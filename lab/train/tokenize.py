# based on https://github.com/KellerJordan/modded-nanogpt/blob/fedb00d76185993997e9921a217562298d1865ee/data/fineweb.py

import os
from dataclasses import dataclass

import numpy as np
import tiktoken
from datasets import load_dataset
from multiprocess.pool import Pool
from simple_parsing import Serializable, field
from tqdm import tqdm


@dataclass
class DatasetConfig(Serializable):
    path: str
    name: str | None = None
    split: str | None = "train"
    text_key: str = "text"


@dataclass
class TokenizerConfig(Serializable):
    encoding_name: str = "gpt2"
    eot_token_str: str = "<|endoftext|>"


@dataclass
class ShardConfig(Serializable):
    magic_number: int = 20240520
    version: int = 1
    num_tokens: int = 10**8


@dataclass
class Config(Serializable):
    cache_dir: str
    dataset: DatasetConfig
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    shard: ShardConfig = field(default_factory=ShardConfig)


def tokens_to_array(tokens: list[int] | np.ndarray, max_int: int = 2**16) -> np.ndarray:
    array = np.array(tokens) if isinstance(tokens, list) else tokens
    assert (array >= 0).all(), "expected all tokens >= 0"
    assert (array < max_int).all(), f"expected all tokens < {max_int}"
    return array.astype(np.uint16)


def tokenize(config: Config) -> None:
    os.makedirs(config.cache_dir, exist_ok=True)

    dataset = load_dataset(
        path=config.dataset.path,
        name=config.dataset.name,
        split=config.dataset.split,
    )

    encoding = tiktoken.get_encoding(config.tokenizer.encoding_name)
    eot_token_id = encoding._special_tokens[config.tokenizer.eot_token_str]

    def _tokenize_row(row: dict[str, str]) -> np.ndarray:
        text = row[config.dataset.text_key]
        assert isinstance(text, str), f"expected str, got {type(text)}"
        tokens = [eot_token_id] + encoding.encode_ordinary(text)
        return tokens_to_array(tokens)

    def _get_pbar(shard_index: int) -> tqdm:
        return tqdm(
            total=config.shard.num_tokens,
            unit="tokens",
            desc=f"shard {shard_index}",
        )

    def _get_shard_filename(shard_index: int) -> str:
        filename = config.dataset.path.replace("/", "_")
        if config.dataset.name is not None:
            filename += f"_{config.dataset.name}"
        if config.dataset.split is not None:
            filename += f"_{config.dataset.split}"
        filename += f"_{shard_index:06d}.bin"
        return os.path.join(config.cache_dir, filename)

    def _write_shard_file(filename: str, buffer: np.ndarray) -> None:
        assert len(buffer) < 2**31, "expected < 2**31 tokens"
        print(f"writing {len(buffer):,} tokens to {filename}")

        header = np.empty((3,), dtype=np.int32)
        header[0] = config.shard.magic_number
        header[1] = config.shard.version
        header[2] = len(buffer)

        with open(filename, "wb") as f:
            f.write(header.tobytes())
            f.write(buffer.tobytes())

    with Pool(min((os.cpu_count() or 1) - 2, 1)) as pool:
        shard_index = 0
        pbar = _get_pbar(shard_index)

        buffer = np.empty((config.shard.num_tokens,), dtype=np.uint16)
        buffer_pos = 0

        for tokens in pool.imap(_tokenize_row, dataset, chunksize=16):  # type: ignore
            num_tokens = len(tokens)

            if buffer_pos + num_tokens < config.shard.num_tokens:
                buffer[buffer_pos : buffer_pos + num_tokens] = tokens
                buffer_pos += num_tokens

                pbar.update(num_tokens)
            else:
                num_tokens_remain = config.shard.num_tokens - buffer_pos
                buffer[buffer_pos : buffer_pos + num_tokens_remain] = tokens[:num_tokens_remain]  # fmt: skip

                _write_shard_file(_get_shard_filename(shard_index), buffer)

                shard_index += 1
                pbar = _get_pbar(shard_index)

                buffer[0 : num_tokens - num_tokens_remain] = tokens[num_tokens_remain:]
                buffer_pos = num_tokens - num_tokens_remain

        if buffer_pos > 0:
            _write_shard_file(_get_shard_filename(shard_index), buffer[:buffer_pos])
            pbar.close()
