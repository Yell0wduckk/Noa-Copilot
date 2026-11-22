"""Install the eight-layer MERT frontend from a local or official snapshot."""

import argparse
import json
import os
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.numpy import save_file

REPOSITORY = "m-a-p/MERT-v2-FullSong"
REVISION = "7001f7657276238629e61c7746ce70735e0604c0"
COMPANION_FILES = (
    "configuration_mert2.py", "modeling_mert2.py", "preprocessor_config.json",
    "LICENSE", "THIRD_PARTY_NOTICES.md",
)


def export(source, output):
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["num_hidden_layers"] = 8
    output.mkdir(parents=True, exist_ok=True)
    with safe_open(source / "model.safetensors", framework="numpy") as weights:
        selected = {
            key: weights.get_tensor(key)
            for key in weights.keys()
            if not key.startswith("layers.") or int(key.split(".", 2)[1]) < 8
        }
        save_file(selected, output / "model.safetensors", metadata=weights.metadata())
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    for name in COMPANION_FILES:
        shutil.copyfile(source / name, output / name)
    (output / "README.md").write_text(
        "# MERT-v2-FullSong L8\n\n"
        "Source: [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong). "
        "This Flat320 export retains the original frontend, subsampler and first eight "
        "Conformer blocks at their original precision. Blocks 9 through 24 were removed. "
        "Weights are licensed under [CC BY-NC 4.0](LICENSE).\n",
        encoding="utf-8",
    )


def download_and_export(output):
    from filelock import FileLock

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / ".mert-v2-install"
    # Keep interrupted transfers next to the portable model for the next attempt.
    os.environ["HF_HOME"] = str(work / "hub-cache")
    from huggingface_hub import hf_hub_download

    with FileLock(output.parent / ".mert-v2-install.lock"):
        source, trimmed = work / "source", work / "trimmed"
        for name in ("config.json", *COMPANION_FILES, "model.safetensors"):
            print(f"正在下载 {name}", flush=True)
            hf_hub_download(REPOSITORY, name, revision=REVISION, local_dir=source, token=False)
        print("正在裁剪前 8 层…", flush=True)
        export(source, trimmed)
        output.mkdir(parents=True, exist_ok=True)
        # Publish the weights last so inference only discovers a complete new frontend.
        for name in ("config.json", *COMPANION_FILES, "README.md", "model.safetensors"):
            (trimmed / name).replace(output / name)
        shutil.rmtree(work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source is None:
        download_and_export(args.output.resolve())
    else:
        export(args.source, args.output)
