"""Stage and publish a verified local copy of the frozen BGE reranker."""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.reranker import (
    MODEL_ID, REVISION, _INVENTORY, _LFS_SHA256, _REQUIRED_FILES, _verify_model,
)
from contracts.index_format import sha256_of


def _download_file(name, cache_dir):
    from huggingface_hub import hf_hub_download

    # Transport logs and exceptions can contain the CDN's signed redirect URL.
    previous_log_threshold = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        cached_path = hf_hub_download(
            repo_id=MODEL_ID, filename=name, revision=REVISION,
            cache_dir=cache_dir, token=False,
        )
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        http_status = f", HTTP {status}" if isinstance(status, int) else ""
        raise RuntimeError(f"download failed for {name} ({type(error).__name__}{http_status})") from None
    finally:
        logging.disable(previous_log_threshold)
    source = Path(cached_path).resolve(strict=True)
    if not source.is_relative_to(cache_dir) or not source.is_file():
        raise ValueError(f"downloaded file is outside the expected cache: {name}")
    return source


def download(out, cache_dir):
    out = Path(out).resolve()
    cache_dir = Path(cache_dir).resolve()
    if cache_dir.is_relative_to(out):
        raise ValueError("the Hugging Face cache must be outside the model output directory")
    if (out / _INVENTORY).exists():
        return _verify_model(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("model output must be absent, empty, or already verified")

    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as staging:
        staged = Path(staging)
        files = {}
        for name in sorted(_REQUIRED_FILES, key=lambda name: (name == "model.safetensors", name)):
            source = _download_file(name, cache_dir)
            destination = staged / name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            files[name] = _LFS_SHA256[name] if name in _LFS_SHA256 else sha256_of(destination)
        inventory = {"model_id": MODEL_ID, "revision": REVISION, "files": files}
        (staged / _INVENTORY).write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        inventory_sha256 = _verify_model(staged)

        if (out / _INVENTORY).exists():
            return _verify_model(out)
        try:
            if out.is_dir():
                out.rmdir()  # Windows cannot rename a directory over an existing one.
            staged.rename(out)
        except OSError:
            if (out / _INVENTORY).exists():
                return _verify_model(out)
            raise
    return inventory_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="verified model output directory")
    parser.add_argument("--cache-dir", required=True, type=Path, help="persistent Hugging Face download cache")
    args = parser.parse_args()
    try:
        inventory_sha256 = download(args.out, args.cache_dir)
    except ImportError:
        parser.exit(1, "Model download requires huggingface-hub.\n")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Model preparation failed: {error}\n")
    print(f"Verified {MODEL_ID} at {REVISION}, inventory SHA256 {inventory_sha256}")


if __name__ == "__main__":
    main()
