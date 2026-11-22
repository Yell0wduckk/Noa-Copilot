"""Probe the imports used by NoaCopilot in the selected inference interpreter."""

import csv
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
from pathlib import Path
import sys


# The chart decoder can also run with audio features produced by another environment.
CHART = {
    "torch": ("torch.nn.functional", ()),
    "numpy": ("numpy", ()),
    "scipy": ("scipy.signal", ("resample_poly",)),
    "soundfile": ("soundfile", ()),
}
AUDIO = {
    "transformers": ("transformers", ("PreTrainedModel", "AutoModel")),
    "torchaudio": ("torchaudio.transforms", ("AmplitudeToDB", "MelScale", "Spectrogram")),
    "huggingface-hub": ("huggingface_hub", ("hf_hub_download",)),
    "safetensors": ("safetensors.torch", ("load_file",)),
    "einops": ("einops", ()),
}


def dll_imports(path):
    """Read the regular PE import table without loading or executing the DLL."""
    import mmap
    import struct
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        pe = struct.unpack_from("<I", data, 0x3c)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return
        sections = struct.unpack_from("<H", data, pe + 6)[0]
        optional_size = struct.unpack_from("<H", data, pe + 20)[0]
        optional = pe + 24
        directories = optional + (112 if struct.unpack_from("<H", data, optional)[0] == 0x20b else 96)
        imports, size = struct.unpack_from("<II", data, directories + 8)
        if not imports:
            return
        table = optional + optional_size
        ranges = [struct.unpack_from("<III", data, table + 40 * i + 12) for i in range(sections)]

        def offset(rva):
            for address, length, raw in ranges:
                if address <= rva < address + length:
                    return raw + rva - address
            raise ValueError("invalid PE section offset")

        start = offset(imports)
        for entry in range(start, min(len(data), start + size), 20):
            name = struct.unpack_from("<I", data, entry + 12)[0]
            if not name:
                break
            pos = offset(name)
            yield data[pos:pos + 260].split(b"\0", 1)[0].decode("ascii")


def unresolved_dlls(dist, error):
    """Follow imports of the binary named in a Windows loader error."""
    import ctypes
    import struct
    names = set(re.findall(r"[\w.+-]+\.(?:dll|pyd)\b", str(error), re.I))
    module = re.search(r"while importing ([\w.]+)", str(error))
    if not (names or module):
        return []
    binaries = [Path(dist.locate_file(path)) for path in dist.files or ()
                if path.suffix.lower() in (".dll", ".pyd")]
    pending = [path for path in binaries if path.name.lower() in {n.lower() for n in names}
               or (module and path.name.split(".")[0] == module[1].split(".")[-1])]
    system = Path(os.environ["SystemRoot"]) / "System32"
    dirs = [Path(sys.executable).parent, Path(sys.base_prefix) / "DLLs",
            Path(dist.locate_file("torch/lib")), system, *map(Path, os.get_exec_path())]
    seen, missing = set(), set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            for name in dll_imports(path):
                if name.lower().startswith(("api-ms-", "ext-ms-")) or ctypes.windll.kernel32.GetModuleHandleW(name):
                    continue
                child = next((d / name for d in (path.parent, *dirs) if (d / name).is_file()), None)
                if child is None:
                    missing.add(name)
                elif child.parent != system:
                    pending.append(child)
        except (OSError, ValueError, struct.error):
            continue
    return sorted(missing)


def probe(check_cuda, expect_cuda_build=False):
    from functools import lru_cache
    try:
        from packaging.requirements import Requirement
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
        except ImportError:
            Requirement = None

    @lru_cache(None)
    def distribution(name):
        try:
            return metadata.distribution(name)
        except metadata.PackageNotFoundError:
            return None

    repair = {}
    reasons = {}
    missing = set()

    def issue(owner, detail):
        reasons.setdefault(owner, [])
        if detail not in reasons[owner]:
            reasons[owner].append(detail)

    def dependency_tree(name, seen):
        if name in seen:
            return
        seen.add(name)
        dist = distribution(name)
        if dist is None:
            missing.add(name)
            issue(name, f"缺少包 {name}")
            repair[name] = ""
            return
        for module in (dist.read_text("top_level.txt") or "").splitlines():
            if module.isidentifier() and importlib.util.find_spec(module) is None:
                issue(name, f"{name} · 缺少模块 {module}")
                repair[name] = dist.version
        if Requirement:
            for raw in dist.requires or ():
                req = Requirement(raw)
                if not req.marker or req.marker.evaluate({"extra": ""}):
                    dependency_tree(req.name.lower().replace("_", "-"), seen)

    def native_files(name):
        dist = distribution(name)
        if not dist or os.name != "nt":
            return []
        # RECORD is supplied by the wheel and also lists delayed-load CUDA DLLs.
        record = dist.read_text("RECORD") or ""
        return [Path(row[0]).name for row in csv.reader(
            line for line in record.splitlines() if ".dll," in line.lower() or ".pyd," in line.lower()
        ) if row and not dist.locate_file(row[0]).is_file()]

    trees = {}
    for name in (*CHART, *AUDIO):
        trees[name] = set()
        dependency_tree(name, trees[name])
        if name not in missing and name in ("torch", "torchaudio"):
            for filename in native_files(name):
                issue(name, f"{name} · 缺少 {filename}")
                repair[name] = distribution(name).version

    for name, (module, symbols) in {**CHART, **AUDIO}.items():
        dependencies = trees[name]
        # These entry points import torch even when wheel metadata makes it optional.
        if name in ("torchaudio", "transformers", "safetensors"):
            dependencies = dependencies | trees["torch"]
        if any(dep in reasons for dep in dependencies):
            continue
        try:
            imported = importlib.import_module(module)
            for symbol in symbols:
                getattr(imported, symbol)
        except Exception as error:
            cause = error
            while cause.__cause__ is not None:
                cause = cause.__cause__
            top = (getattr(cause, "name", "") or "").split(".")[0].lower().replace("_", "-")
            owner = top if top in dependencies else name
            detail = str(error)
            if cause is not error and str(cause) not in detail:
                detail += f" ({cause})"
            if os.name == "nt":
                for filename in unresolved_dlls(distribution(name), cause):
                    issue(name, f"{name} · 找不到依赖 DLL {filename}")
            issue(name, f"{name} · {detail}")
            if owner not in missing:
                repair[owner] = distribution(owner).version
        if name in reasons and name not in ("torch", "torchaudio"):
            for filename in native_files(name):
                issue(name, f"{name} · 缺少 {filename}")
                repair[name] = distribution(name).version

    torch = sys.modules.get("torch")
    if expect_cuda_build and torch is not None and not torch.version.cuda:
        issue("torch", "torch · 当前安装的是 CPU 版，缺少 CUDA 版")
        repair["torch"] = distribution("torch").version
    torchaudio = sys.modules.get("torchaudio")
    if expect_cuda_build and torchaudio is not None and "+cpu" in torchaudio.__version__:
        issue("torchaudio", "torchaudio · 当前安装的是 CPU 版，缺少 CUDA 版")
        repair["torchaudio"] = distribution("torchaudio").version

    def messages(dependencies):
        return [message for name in sorted(dependencies) for message in reasons.get(name, ())]

    chart, audio = (messages(set().union(*(trees[name] for name in checks))) for checks in (CHART, AUDIO))
    gpu = cuda_error = ""
    if check_cuda and torch is not None and not chart:
        try:
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except Exception as error:
            cuda_error = str(error)
    return dict(chart=chart, audio=audio, repair=repair, gpu=gpu, cuda_error=cuda_error,
                version=str(getattr(torch, "__version__", "")))


if __name__ == "__main__":
    print(json.dumps(probe("cuda" in sys.argv[1:], "cuda-build" in sys.argv[1:])))
