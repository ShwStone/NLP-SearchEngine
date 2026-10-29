"""Download a pinned model and verify a local model inventory."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen

from contracts.index_format import sha256_of
from .embedding import bge_spec

FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt")
PINNED_SHA256 = {
    "config.json": "3853a7979202c348751b753e36f579c41d8da7d36af617d3d907e1fc9b441f2a",
    "model.safetensors": "354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026",
    "tokenizer.json": "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
    "tokenizer_config.json": "e6f3b96db926a37d4039995fbf5ad17de158dfb8f6343d607e4dbaad18d75f5a",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}


def fetch(url, target, proxy=None):
    if shutil.which("curl"):
        command = ["curl", "-fL", "--retry", "3", "--retry-all-errors", "--max-time", "240"]
        if proxy:
            command += ["--proxy", proxy, "--noproxy", ""]
        subprocess.run(command + [url, "-o", str(target)], check=True)
    else:
        with urlopen(url, timeout=120) as response, Path(target).open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)


def verify_model(directory, spec):
    root = Path(directory)
    inventory = json.loads((root / "model_inventory.json").read_text(encoding="utf-8"))
    if inventory["model_id"] != spec.model_id or inventory["revision"] != spec.revision:
        raise ValueError("local model revision does not match embedding spec")
    for name in FILES:
        if inventory["files"].get(name) != PINNED_SHA256[name] or sha256_of(root / name) != PINNED_SHA256[name]:
            raise ValueError(f"local model checksum mismatch: {name}")


def download(directory, proxy=None):
    spec = bge_spec()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "model_inventory.json").is_file():
        verify_model(root, spec)
        return
    for name in FILES:
        url = f"https://huggingface.co/{spec.model_id}/resolve/{spec.revision}/{name}"
        pending = root / (name + ".part")
        fetch(url, pending, proxy)
        if sha256_of(pending) != PINNED_SHA256[name]:
            raise ValueError(f"downloaded model checksum mismatch: {name}")
        pending.replace(root / name)
    inventory = dict(model_id=spec.model_id, revision=spec.revision,
                     files={name: sha256_of(root / name) for name in FILES})
    (root / "model_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    download(args.out, args.proxy)
