from lab.components.feed_forward.swiglu import SwiGLU

from ..model import ModelConfig

FLOPS_SQRT = 8
FLOPS_EXP = 10
FLOPS_SOFTMAX = FLOPS_EXP + 2
FLOPS_SILU = FLOPS_EXP + 4


def flops_rms_norm(config: ModelConfig) -> int:
    return config.max_seq_len * (4 * config.dim + 2 + FLOPS_SQRT)


def get_hidden_dim(config: ModelConfig) -> int:
    swiglu = SwiGLU(
        config.dim, 4 * config.dim, config.multiple_of, config.ffn_dim_multiplier
    )
    return swiglu.w1.out_features


def flops_swiglu(config: ModelConfig) -> int:
    hidden_dim = get_hidden_dim(config)
    flops = 0
    # matmul: self.w1(x)
    flops += 2 * config.max_seq_len * config.dim * hidden_dim
    # matmul: self.w2(...)
    flops += 2 * config.max_seq_len * hidden_dim * config.dim
    # matmul: self.w3(x)
    flops += 2 * config.max_seq_len * config.dim * hidden_dim
    # elementwise func: F.silu(...)
    flops += config.max_seq_len * hidden_dim * FLOPS_SILU
    # elementwise mult: F.silu(...) * self.w3(x)
    flops += config.max_seq_len * hidden_dim
    return flops


def flops_apply_rotary_emb(config: ModelConfig) -> int:
    flops = 0
    # complex mult: xq_ * freqs_cis
    flops += 6 * config.max_seq_len * config.n_heads * config.head_dim // 2
    # complex mult: xk_ * freqs_cis
    flops += 6 * config.max_seq_len * config.n_kv_heads * config.head_dim // 2  # type: ignore
    return flops


def flops_grouped_query_attention(config: ModelConfig) -> int:
    flops = 0
    # matmul: self.wq(xq)
    flops += 2 * config.max_seq_len * config.dim * config.n_heads * config.head_dim
    # matmul: self.wk(xk)
    flops += 2 * config.max_seq_len * config.dim * config.n_kv_heads * config.head_dim  # type: ignore
    # matmul: self.wv(xv)
    flops += 2 * config.max_seq_len * config.dim * config.n_kv_heads * config.head_dim  # type: ignore
    # apply_rotary_emb(...)
    flops += flops_apply_rotary_emb(config)
    # matmul: torch.matmul(xq, xk...)
    flops += 2 * config.max_seq_len * config.max_seq_len * config.dim
    # elementwise div: ... / math.sqrt(self.head_dim)
    flops += config.n_heads * config.max_seq_len * config.max_seq_len
    # elementwise add: scores + mask
    flops += config.n_heads * config.max_seq_len * config.max_seq_len
    # F.softmax(...)
    flops += config.n_heads * config.max_seq_len * config.max_seq_len * FLOPS_SOFTMAX
    # matmul: torch.matmul(scores, xv)
    flops += 2 * config.max_seq_len * config.max_seq_len * config.dim
    # matmul: self.wo(output)
    flops += 2 * config.max_seq_len * config.dim * config.dim
    return flops


def flops_transformer_block(config: ModelConfig) -> int:
    flops = 0
    # self.attention(...)
    flops += flops_grouped_query_attention(config)
    # self.feed_forward(...)
    flops += flops_swiglu(config)
    # self.pre_attention_norm(...)
    flops += flops_rms_norm(config)
    # self.post_attention_norm(...)
    flops += flops_rms_norm(config)
    # self.pre_feed_forward_norm(...)
    flops += flops_rms_norm(config)
    # self.post_feed_forward_norm(...)
    flops += flops_rms_norm(config)
    # elementwise mult: gate * self.post_attention_norm...
    flops += config.max_seq_len * config.dim
    # elementwise add: x = x + ...
    flops += config.max_seq_len * config.dim
    # elementwise mult: gate * self.post_feed_forward_norm...
    flops += config.max_seq_len * config.dim
    # elementwise add: return x + gate ...
    flops += config.max_seq_len * config.dim
    return flops


def flops_transformer(
    config: ModelConfig, sparse: bool = False, sparsity: float = 0
) -> int:
    flops = 0
    # lookup: self.tok_embeddings(inputs)
    flops += 0
    # self.input_norm(...)
    flops += flops_rms_norm(config)
    if sparse:
        # matmul: self.masks[i](h)
        flops += 2 * (0.5 * config.n_layers) * config.max_seq_len * config.dim
        # elementwise add: ... + (masks[-1] if ...
        flops += (0.5 * config.n_layers - 1) * config.max_seq_len
    # layer(h, ...)
    flops += 2 * (1 - sparsity) * config.n_layers * flops_transformer_block(config)
    # self.output_norm(h)
    flops += flops_rms_norm(config)
    # matmul: self.output(...)
    flops += 2 * config.max_seq_len * config.dim * config.vocab_size
    return int(flops)
