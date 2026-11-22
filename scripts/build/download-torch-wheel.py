"""Fetch Windows PyTorch wheels from the selected mirrors."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
from threading import Event
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from zipfile import is_zipfile


def download(url, path):
    def request(first, last):
        return urlopen(Request(url, headers={"Range": f"bytes={first}-{last}"}), timeout=30)

    with request(0, 0) as response:
        if response.status == 200:
            with path.open("wb") as output:
                shutil.copyfileobj(response, output, 1 << 20)
            return
        total = int(response.headers["Content-Range"].split("/")[-1])

    with path.open("wb") as output:
        output.truncate(total)

    stopped = Event()

    def fetch(first, end):
        position = first
        for attempt in range(3):
            try:
                with request(position, end - 1) as response, path.open("r+b") as output:
                    expected = f"bytes {position}-{end - 1}/{total}"
                    if response.headers.get("Content-Range") != expected:
                        raise OSError("Unexpected byte range from PyTorch CDN")
                    output.seek(position)
                    while position < end:
                        if stopped.is_set():
                            return 0
                        data = response.read(min(1 << 20, end - position))
                        if not data:
                            raise OSError("Incomplete wheel download")
                        output.write(data)
                        position += len(data)
                return end - first
            except HTTPError:
                raise
            except OSError:
                if attempt == 2:
                    raise

    chunk = 8 << 20
    received = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch, first, min(first + chunk, total)) for first in range(0, total, chunk)]
        try:
            for job in as_completed(jobs):
                received += job.result()
                print(f"Progress {received} of {total}", flush=True)
        finally:
            stopped.set()
            for job in jobs:
                job.cancel()


def fetch_wheels(flavor, sources, names=("torch", "torchaudio")):
    for name in names:
        path = Path(f"{name}-2.11.0+{flavor}-cp313-cp313-win_amd64.whl")
        for index, source in enumerate(sources):
            print(f"正在下载 {name}（{urlsplit(source).hostname}）", flush=True)
            try:
                download(f"{source}/{flavor}/{quote(path.name)}", path)
                if not is_zipfile(path):
                    raise OSError(f"{urlsplit(source).hostname} 返回的不是 wheel 文件")
                break
            except (OSError, ValueError, KeyError) as error:
                if index + 1 == len(sources):
                    raise
                print(f"{error}；尝试下一个下载源", flush=True)


if __name__ == "__main__":
    fetch_wheels(sys.argv[1], sys.argv[3:], sys.argv[2].split(','))
