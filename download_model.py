"""Download the fine-tuned adapter into models/.

    python download_model.py           skip if it is already there
    python download_model.py --force   download again

The adapter is published as a GitHub Release asset because it is too big for
the repository itself.
"""

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RELEASE = "https://github.com/lgw123456789/Malay-Rojak-Lemmatiser/releases/download/v1.0"
ADAPTERS = ["lemmatizer_lora_llama"]


def fetch(url, dest):
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print("\r  %3d%%  %d / %d MB" % (100 * done // total, done >> 20, total >> 20),
                      end="", flush=True)
    print()


def install(name, force):
    target = MODELS_DIR / name
    if (target / "adapter_model.safetensors").exists() and not force:
        print("%s is already in models/ - nothing to do." % name)
        return True

    url = "%s/%s.zip" % (RELEASE, name)
    print("downloading %s" % url)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / (name + ".zip")
        try:
            fetch(url, archive)
        except Exception as exc:
            print("  download failed: %s" % exc)
            return False
        with zipfile.ZipFile(archive) as z:
            bad = [n for n in z.namelist() if n.startswith("/") or ".." in Path(n).parts]
            if bad:
                print("  unexpected paths in the archive, not extracting")
                return False
            if target.exists():
                shutil.rmtree(target)
            z.extractall(MODELS_DIR)

    if not (target / "adapter_model.safetensors").exists():
        print("  the archive did not contain %s/adapter_model.safetensors" % name)
        return False
    print("installed -> models/%s" % name)
    return True


def main():
    p = argparse.ArgumentParser(description="Download the fine-tuned adapter.")
    p.add_argument("--force", action="store_true", help="download even if present")
    a = p.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    ok = all(install(name, a.force) for name in ADAPTERS)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
