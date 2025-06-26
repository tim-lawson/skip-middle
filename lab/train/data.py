import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from simple_parsing import Serializable


@dataclass
class DataConfig(Serializable):
    train_files: str
    train_tokens: int | None
    val_files: str
    val_tokens: int | None
    batch_size: int
    device_batch_size: int


# based on https://github.com/KellerJordan/modded-nanogpt/blob/822ab2dd79140ed34ae43a20450f0bdc36457a24/records/111024_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt#L279-L352
@dataclass
class DataHeader:
    MAGIC_NUMBER = 20240520
    VERSION = 1
    HEADER_SIZE = 256 * 4  # 256 int32 bytes

    magic_number: int
    version: int
    num_tokens: int

    @classmethod
    def from_bytes(cls, buffer: bytes) -> "DataHeader":
        header = np.frombuffer(buffer, dtype=np.int32)
        return cls(magic_number=header[0], version=header[1], num_tokens=int(header[2]))

    def validate(self) -> None:
        if self.magic_number != self.MAGIC_NUMBER:
            raise ValueError(
                "Invalid magic number. "
                f"Expected {self.MAGIC_NUMBER}, found {self.magic_number}"
            )
        if self.version != self.VERSION:
            raise ValueError(
                "Invalid version. "  #
                f"Expected {self.VERSION}, found {self.version}"
            )


class DataShard:
    def __init__(self, filename: str) -> None:
        self.filename = Path(filename)
        self._tokens: np.ndarray | None = None
        self._header: DataHeader | None = None

    @property
    def header(self) -> DataHeader:
        if self._header is None:
            with open(self.filename, "rb") as f:
                header_bytes = f.read(DataHeader.HEADER_SIZE)
                self._header = DataHeader.from_bytes(header_bytes)
                self._header.validate()
        return self._header

    @property
    def tokens(self) -> np.ndarray:
        if self._tokens is None:
            with open(self.filename, "rb") as f:
                f.seek(DataHeader.HEADER_SIZE)
                self._tokens = np.frombuffer(f.read(), dtype=np.uint16)
                if len(self._tokens) != self.header.num_tokens:
                    raise ValueError(
                        "Token count mismatch. "
                        f"Header {self.header.num_tokens}, found {len(self._tokens)}"
                    )
        return self._tokens

    @property
    def num_tokens(self) -> int:
        return self.header.num_tokens


class DistributedDataLoader:
    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        seq_len: int,
        rank: int,
        world_size: int,
        max_tokens: int | np.int64 | None = None,
        cuda: bool = True,
    ) -> None:
        self.filename_pattern = filename_pattern
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.max_tokens = max_tokens
        self.cuda = cuda

        self.filenames = sorted(glob.glob(filename_pattern))
        if not self.filenames:
            raise FileNotFoundError(
                f"Found no files matching the pattern {filename_pattern}"
            )

        self.files: list[DataShard] = []
        self.min_shard_tokens = self.world_size * self.batch_size * self.seq_len + 1
        self.max_shard_tokens = np.int64(0)
        for filename in self.filenames:
            if self.max_tokens is not None and self.max_shard_tokens >= self.max_tokens:
                break
            shard = DataShard(filename)
            if shard.num_tokens < self.min_shard_tokens:
                raise ValueError(
                    f"Shard {shard.filename} has too few tokens. "
                    f"Need at least {self.min_shard_tokens}, found {shard.num_tokens}"
                )
            self.files.append(shard)
            self.max_shard_tokens += shard.num_tokens

        self.reset()

    def reset(self) -> None:
        self.position = self.rank * self.batch_size * self.seq_len
        self.shard_idx = 0
        self.shard = self.files[self.shard_idx]

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.shard.tokens
        window_size = self.batch_size * self.seq_len

        buffer = tokens[self.position : self.position + window_size + 1]
        buffer = torch.tensor(buffer.astype(np.int32), dtype=torch.long)

        inputs = buffer[:-1].view(self.batch_size, self.seq_len)
        labels = buffer[1:].view(self.batch_size, self.seq_len)

        self.position += window_size * self.world_size
        if self.position + (window_size * self.world_size + 1) > len(tokens):
            self.position = self.rank * self.batch_size * self.seq_len
            self.shard_idx = (self.shard_idx + 1) % len(self.files)
            self.shard = self.files[self.shard_idx]

        if self.cuda:
            return inputs.cuda(), labels.cuda()
        return inputs, labels


def get_steps(
    files: str,
    device_batch_size: int,
    max_seq_len,
    batch_tokens: int,
    max_tokens: int | None = None,
) -> int:
    if max_tokens is not None:
        return int(max_tokens / batch_tokens)
    loader = DistributedDataLoader(files, device_batch_size, max_seq_len, 0, 1)
    return int(loader.max_shard_tokens / batch_tokens)
