from os import PathLike
from typing import Any

__version__: str

def core_version() -> str: ...
def hwinfo_memory() -> dict[str, Any]: ...
def hwinfo_cpu() -> dict[str, Any]: ...
def hwinfo_disk(path: str | PathLike[str]) -> dict[str, Any]: ...

# E008 prototypes (bytes input, RS2 violation); removed when the spill engine lands (0027).
def codec_compress(
    data: bytes, itemsize: int, level: int = 1, chunk_bytes: int = ..., threads: int = 0
) -> list[bytes]: ...
def codec_decompress(
    chunks: list[bytes],
    itemsize: int,
    total_len: int,
    chunk_bytes: int = ...,
    threads: int = 0,
) -> bytes: ...
def codec_compress_reuse_size(
    data: bytes, itemsize: int, level: int = 1, chunk_bytes: int = ..., threads: int = 0
) -> int: ...
def codec_spill_to_file(
    data: bytes,
    itemsize: int,
    path: str,
    level: int = 1,
    chunk_bytes: int = ...,
    threads: int = 0,
) -> int: ...
