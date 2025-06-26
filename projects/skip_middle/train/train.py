import os
import time

import numpy as np
import torch
import torch.distributed as dist
import wandb

from lab.train.data import DistributedDataLoader
from lab.train.schedulers import get_linear_cosine_scheduler
from lab.train.utils import initialize_ddp, initialize_logs, seed_everything
from lab.utils import cross_entropy

from ..gate_controller import get_gate_controller
from ..gate_metrics import GateMetrics
from ..model import Transformer, get_optimizers
from .config import TrainConfig

autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)  # type: ignore


def train(config: TrainConfig) -> None:
    seed_everything(config.seed, cudnn=False)
    ddp = initialize_ddp()
    logdir, run = initialize_logs(config.__dict__, ddp.is_main_process)

    if ddp.is_main_process:
        print(config.dumps_yaml())
        print(f"val_tokens:{config.val_steps * config.minibatch_tokens}")

    model = Transformer(config.model, device=ddp.device)
    model = model.to(ddp.device)
    model.apply(model._init_weights)

    if config.log_gradients and ddp.is_main_process:
        run.watch(model, log="gradients", log_freq=config.log_every_steps)  # type: ignore

    adamw_optimizer = get_optimizers(model, config.optimizer)
    optimizers = [adamw_optimizer]

    schedulers = [
        get_linear_cosine_scheduler(optimizer, config.scheduler, config.train_steps)
        for optimizer in optimizers
    ]

    controller = get_gate_controller(config.gates, config.model.n_layers, ddp.device)

    train_loader = DistributedDataLoader(
        config.data.train_files,
        config.data.device_batch_size,
        config.model.max_seq_len,
        ddp.rank,
        ddp.world_size,
    )
    train_loader.reset()
    train_time_ms = 0
    torch.cuda.synchronize()  # start clock
    t0 = time.perf_counter()  # reset timer

    for step in range(config.train_steps + 1):
        is_last_step = step == config.train_steps

        if is_last_step or (
            config.val_every_steps > 0 and step % config.val_every_steps == 0
        ):
            torch.cuda.synchronize()  # stop clock
            train_time_ms += 1000 * (time.perf_counter() - t0)  # update timer
            model.eval()

            val_loader = DistributedDataLoader(
                config.data.val_files,
                config.data.device_batch_size,
                config.model.max_seq_len,
                ddp.rank,
                ddp.world_size,
            )
            val_loss = torch.zeros(1, device=ddp.device)

            val_metrics = None
            if not config.model.zero_init_masks:
                val_metrics = GateMetrics(
                    config.model.n_layers,
                    config.gates_zero_eps,
                    ddp.device,
                    ddp.is_distributed,
                )

            with torch.no_grad(), autocast:
                for _ in range(config.val_steps):
                    inputs, labels = val_loader.next_batch()
                    logits, info = model(inputs)
                    val_loss += cross_entropy(logits, labels)

                    if val_metrics is not None:
                        val_metrics.update(info.masks, info.gates, info.hidden_states)

            del val_loader
            val_loss /= config.val_steps

            if val_metrics is not None:
                val_metrics.compute()

            if ddp.is_main_process:
                print(
                    f"step:{step}/{config.train_steps} "  #
                    f"val_loss:{val_loss.item():.4f}"
                )
                metrics = {"val/loss": val_loss.item()}
                if val_metrics is not None:
                    metrics.update(val_metrics.to_dict("val/"))
                wandb.log(metrics, step=step)

            model.train()
            torch.cuda.synchronize()  # start clock
            t0 = time.perf_counter()  # reset timer

        if is_last_step or (
            config.save_every_steps > 0 and step % config.save_every_steps == 0
        ):
            torch.cuda.synchronize()  # stop clock
            train_time_ms += 1000 * (time.perf_counter() - t0)  # update timer

            if ddp.is_main_process:
                state = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                }
                state_name = f"state_step{step}.pt"
                state_path = os.path.join(logdir, state_name)

                torch.save(state, state_path)  # type: ignore

                with open(os.path.join(logdir, "config.json"), "w") as f:  # type: ignore
                    f.write(config.dumps_json())

                artifact = wandb.Artifact(name="state", type="model")
                artifact.add_file(local_path=state_path, name=state_name)
                artifact.save()

            torch.cuda.synchronize()  # start clock
            t0 = time.perf_counter()  # reset timer

        if is_last_step:
            break

        train_loss = torch.zeros(1, device=ddp.device)

        train_metrics = None
        if not config.model.zero_init_masks:
            train_metrics = GateMetrics(
                config.model.n_layers,
                config.gates_zero_eps,
                ddp.device,
                ddp.is_distributed,
            )

        for _ in range(config.grad_acc_steps):
            with autocast:
                inputs, labels = train_loader.next_batch()
                logits, info = model(inputs)
                loss = cross_entropy(logits, labels)
                loss += controller.loss(info.gates)

            loss.backward()

            train_loss += loss.detach()

            if train_metrics is not None:
                train_metrics.update(info.masks, info.gates, info.hidden_states)

        for param in model.parameters():
            if ddp.is_distributed:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            param.grad /= config.grad_acc_steps  # type: ignore

        for optimizer in optimizers:
            optimizer.step()

        for scheduler in schedulers:
            scheduler.step()

        model.zero_grad(set_to_none=True)

        if ddp.is_distributed:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        train_loss /= config.grad_acc_steps

        if train_metrics is not None:
            train_metrics.compute()
            controller.update(train_metrics.gates_mean, train_metrics.gates_var)

        approx_train_time_ms = train_time_ms + 1000 * (time.perf_counter() - t0)
        if ddp.is_main_process:
            print(
                f"step:{step}/{config.train_steps} "
                f"train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms "
                f"step_avg:{approx_train_time_ms / (step + 1):.2f}ms"
            )

            if config.log_every_steps > 0 and step % config.log_every_steps == 0:
                metrics = {
                    "train/num_tokens": step * config.batch_tokens,
                    "train/loss": train_loss.item(),
                    "train/lr/default": adamw_optimizer.param_groups[0]["lr"],
                    "train/lr/masks": adamw_optimizer.param_groups[1]["lr"],
                    "train/lr/norms": adamw_optimizer.param_groups[2]["lr"],
                }
                if train_metrics is not None:
                    metrics.update(train_metrics.to_dict("train/"))
                    metrics.update(controller.to_dict("train/controller/"))
                wandb.log(metrics, step=step)

                if config.log_params:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            hist = np.histogram(param.detach().cpu().float().numpy())
                            wandb.log(
                                {f"params/{name}": wandb.Histogram(np_histogram=hist)},
                                step=step,
                            )

    if ddp.is_main_process:
        max_memory_allocated_mib = torch.cuda.max_memory_allocated() // 1024 // 1024
        print(f"max_memory_allocated {max_memory_allocated_mib}MiB")

        wandb.finish()
