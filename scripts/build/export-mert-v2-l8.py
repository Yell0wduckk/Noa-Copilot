"""Export the native Flat320 MERT-v2-FullSong frontend from a local HF snapshot."""

import argparse
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.numpy import save_file


def export(source, output):
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["num_hidden_layers"] = 4
    output.mkdir(parents=True, exist_ok=True)
    with safe_open(source / "model.safetensors", framework="numpy") as weights:
        selected = {
            key: weights.get_tensor(key)
            for key in weights.keys()
            if not key.startswith("layers.") or int(key.split(".", 2)[1]) < 4
        }
        save_file(selected, output / "model.safetensors", metadata=weights.metadata())
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    for name in (
        "configuration_mert2.py", "modeling_mert2.py", "preprocessor_config.json",
        "LICENSE", "THIRD_PARTY_NOTICES.md",
    ):
        shutil.copyfile(source / name, output / name)
    (output / "README.md").write_text(
        "# MERT-v2-FullSong L4\n\n"
        "Source: [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong). "
        "This Flat320 export retains the original frontend, subsampler and first four "
        "Conformer blocks at their original precision. Blocks 5 through 24 were removed. "
        "Weights are licensed under [CC BY-NC 4.0](LICENSE).\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
