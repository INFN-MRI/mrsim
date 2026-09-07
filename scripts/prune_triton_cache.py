"""Reclaim the disk a Triton cache grows into, without losing a compilation.

Two things make the cache grow without bound. Each entry keeps the MLIR the
kernel was compiled from -- a `.source` file carrying a source location per
operation, which on a kernel the size of the EPG state machine is most of the
entry and is never read again: a launch reads the cubin and its metadata.
And nothing evicts: Triton keys an entry on the text of the file the kernel
was written in, so every edit to ``_epg_triton.py`` orphans every entry made
before it, and the orphans stay.

So there are two prunes here. Dropping the IR is free -- the entries still
answer, and nothing recompiles. Dropping entries by age is not: an entry still
in use recompiles the next time it is asked for, which is the ordinary cold
cost of that kernel.

Nothing is deleted without ``--apply``, so the first two commands below only
say what the third would do.

    python scripts/prune_triton_cache.py
    python scripts/prune_triton_cache.py --drop-ir --older-than 30
    python scripts/prune_triton_cache.py --drop-ir --older-than 30 --apply

``--cache`` names the directory; without it, ``TRITON_CACHE_DIR`` and then
``~/.triton/cache``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import time

#: What a launch reads. Everything else in an entry is there for a human.
LAUNCHED_FROM = {".cubin", ".hsaco", ".json", ".so"}


def cache_directory(named: str | None) -> pathlib.Path:
    """The cache to work on, from the argument, the environment, or the default."""
    if named:
        return pathlib.Path(named).expanduser()
    from_environment = os.environ.get("TRITON_CACHE_DIR")
    if from_environment:
        return pathlib.Path(from_environment).expanduser()
    return pathlib.Path.home() / ".triton" / "cache"


def entries(cache: pathlib.Path) -> list[pathlib.Path]:
    """The per-kernel directories, each named by its compilation's hash."""
    return sorted(path for path in cache.iterdir() if path.is_dir())


def bytes_under(path: pathlib.Path) -> int:
    """Total size of everything below ``path``."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def intermediate_ir(entry: pathlib.Path) -> list[pathlib.Path]:
    """The files in ``entry`` that no launch reads."""
    return [
        file
        for file in entry.iterdir()
        if file.is_file() and file.suffix not in LAUNCHED_FROM
    ]


def touched(entry: pathlib.Path) -> float:
    """When anything in ``entry`` was last read or written."""
    return max(
        (file.stat().st_atime for file in entry.rglob("*") if file.is_file()),
        default=entry.stat().st_atime,
    )


def main() -> None:
    """Report what the cache holds, and drop what was asked for."""
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--cache", default="", help="the cache directory")
    cli.add_argument(
        "--drop-ir",
        action="store_true",
        help="delete the MLIR no launch reads",
    )
    cli.add_argument(
        "--older-than",
        type=float,
        default=0.0,
        help="entries untouched for this many days, which recompile if asked for again",
    )
    cli.add_argument(
        "--apply",
        action="store_true",
        help="delete what the others select; without it nothing is touched",
    )
    arguments = cli.parse_args()

    cache = cache_directory(arguments.cache)
    if not cache.is_dir():
        print(f"no cache at {cache}")
        return

    held = entries(cache)
    total = sum(bytes_under(entry) for entry in held)
    print(f"{cache}: {len(held)} entries, {total / 2**30:.2f} GiB")

    stale = []
    if arguments.older_than > 0:
        cutoff = time.time() - arguments.older_than * 86400
        stale = [entry for entry in held if touched(entry) < cutoff]

    ir = {entry: intermediate_ir(entry) for entry in held if entry not in set(stale)}
    ir_bytes = sum(file.stat().st_size for files in ir.values() for file in files)
    stale_bytes = sum(bytes_under(entry) for entry in stale)

    print(f"  IR no launch reads: {ir_bytes / 2**30:.2f} GiB")
    if arguments.older_than > 0:
        print(
            f"  entries untouched for {arguments.older_than:g} days: "
            f"{len(stale)}, {stale_bytes / 2**30:.2f} GiB"
        )

    if not arguments.apply:
        print("nothing deleted; pass --apply to act on the above")
        return

    reclaimed = 0
    for entry in stale:
        reclaimed += bytes_under(entry)
        shutil.rmtree(entry)
    if arguments.drop_ir:
        for files in ir.values():
            for file in files:
                reclaimed += file.stat().st_size
                file.unlink()

    remaining = sum(bytes_under(entry) for entry in entries(cache))
    print(f"reclaimed {reclaimed / 2**30:.2f} GiB, {remaining / 2**30:.2f} GiB left")


if __name__ == "__main__":
    main()
