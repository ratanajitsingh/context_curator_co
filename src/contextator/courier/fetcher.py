from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FetchResult:
    src_path: str
    content: str | None
    success: bool
    error: str | None = None
    size_bytes: int = 0
    fetched_at: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        status = "ok" if self.success else f"Fucken failed man shiba ({self.error})"
        return f"FetchResult({self.src_path!r}, {status}, {self.size_bytes}B)"


#adding a max file size not to crash anything accidentally - currently put it to be 5mb should change it to higher later
MAX_FILE_SIZE_BYTES = 5_000_000

async def fetch_file(path:str | Path) -> FetchResult:
    #thisll fetch a single files content, threaded it to make it parrallelizable or however you spell it
    path = Path(path)
    def _read() -> FetchResult:
        if not path.exists():
            return FetchResult(
                src_path= str(path),
                content=None,
                success=False,
                error="no file found",
            )
        if not path.is_file():
            return FetchResult(
                src_path=str(path),
                content = None,
                success = False,
                error = "Not a path to a file",
            )

        size = path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            return FetchResult(
                src_path = str(path),
                content = None,
                success = False,
                error = "File is too large",
                size_bytes= size
            )
        try:
            text = path.read_text(encoding = "utf-8")
        except UnicodeDecodeError as e:
            return FetchResult(
                src_path=str(path),
                content = None,
                success = False,
                error = f"Couldn't decode as utf-8: {e}",
                size_bytes=size,
            )
        except OSError as e:
            return FetchResult(
                src_path = str(path),
                content = None,
                success = False,
                error = f"os error : {e}",
                size_bytes= size,
            )
        #hopefully i have covered all the error cases, will prolly add more if theyre found when testing

        return FetchResult(
            src_path= str(path),
            content=text,
            success = True,
            size_bytes=size,
        )

    return await asyncio.to_thread(_read)

async def fetch_files(paths: list[str | Path]) -> list[FetchResult]:
    #thisll be able to fetch multiple files concurrently, making it independent queries so they dont cancel one another if it fails and planner decides what to do w partial results (out of 5 files only 3 fetched)

    if not paths:
        return []
    return await asyncio.gather(*(fetch_file(p) for p in paths))

def split_successes(results:list[FetchResult]) -> tuple[list[FetchResult], list[FetchResult]]:
    #claude told me to put this statistic for convenience idk
    ok = [r for r in results if r.success]
    no = [r for r in results if not r.success]
    return ok,no


