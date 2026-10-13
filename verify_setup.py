"""Check that the lemmatiser is installed and working.

Usage:
    python verify_setup.py          # app and rules
    python verify_setup.py --gpu    # also require the fine-tuned model packages

Exits 0 when `streamlit run app.py` will work, 1 otherwise.
"""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

REQUIRED = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("openpyxl", "openpyxl"),
]

OPTIONAL = [
    ("enchant", "pyenchant"),
    ("requests", "requests"),
]

GPU_PACKAGES = [
    ("torch", "torch"),
    ("unsloth", "unsloth"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("bitsandbytes", "bitsandbytes"),
    ("triton", "triton-windows"),
]

DATA_FILES = [
    "malindo_morph/malindo_dic_2024.tsv",
    "malindo_morph/BM_dict.csv",
    "data/corpus.jsonl",
    "data/gold.jsonl",
    "data/val.jsonl",
    "data/silver.jsonl",
    "data/sft/train.jsonl",
]

SAMPLE = "Aku dh xleh handle keje tu, Kak Uji pun tau"


class Reporter:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def section(self, title):
        print(f"\n{title}\n{'-' * len(title)}")

    def ok(self, message):
        print(f"{PASS} {message}")

    def warn(self, message):
        print(f"{WARN} {message}")
        self.warnings.append(message)

    def fail(self, message):
        print(f"{FAIL} {message}")
        self.failures.append(message)


def check_python(r):
    r.section("Python")
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if v < (3, 10):
        r.fail(f"{label} - 3.10 or newer is required")
    elif v >= (3, 13):
        r.warn(f"{label} - the GPU packages only support 3.10-3.12")
    else:
        r.ok(label)


def check_packages(r, packages, required=True):
    for module, pip_name in packages:
        try:
            mod = importlib.import_module(module)
        except Exception as e:
            if required:
                r.fail(f"{pip_name} cannot be imported: {e}")
            else:
                r.warn(f"{pip_name} not available: {e}")
            continue
        r.ok(f"{pip_name} {getattr(mod, '__version__', '')}".strip())


def check_data(r):
    r.section("Data files")
    for rel in DATA_FILES:
        path = ROOT / rel
        if path.exists():
            r.ok(f"{rel} ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            r.fail(f"{rel} is missing")


def check_rules(r):
    r.section("Rule pipeline")
    try:
        import pipeline
        result = pipeline.lemmatise(SAMPLE)[0]
    except Exception as e:
        r.fail(f"could not run the rules: {type(e).__name__}: {e}")
        return
    r.ok(f'"{SAMPLE}"')
    r.ok(f'-> "{result["sentence"]}"')


def check_models(r):
    r.section("Fine-tuned adapters (models/)")
    import pipeline
    found = pipeline.adapters_on_disk()
    if found:
        for name in found:
            r.ok(f"{name}: {pipeline.ADAPTERS[name].relative_to(ROOT)}")
    else:
        print("[INFO] none yet - train one with `python train.py --model llama`, "
              "or copy an adapter folder into models/")


def check_gpu(r, required):
    r.section("GPU packages")
    missing = []
    for module, pip_name in GPU_PACKAGES:
        if module == "triton" and sys.platform != "win32":
            continue
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            missing.append(pip_name)
    if missing:
        if required:
            r.fail("not installed: " + ", ".join(missing))
        else:
            print("[INFO] not installed: " + ", ".join(missing))
            print("[INFO] Optional. The rules work without them.")
        return

    try:
        import torch
    except Exception as e:
        r.fail(f"torch cannot be imported: {e}")
        return
    r.ok(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        if required:
            r.fail("CUDA is not available - check the NVIDIA driver")
        else:
            r.warn("CUDA is not available - check the NVIDIA driver")
        return
    props = torch.cuda.get_device_properties(0)
    r.ok(f"{props.name}, {props.total_memory / 2**30:.1f} GB VRAM")
    if props.total_memory < 3.5 * 2**30:
        r.warn("less than 4 GB of VRAM - the model may not fit")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true",
                   help="treat missing GPU packages as failures")
    a = p.parse_args()

    r = Reporter()
    check_python(r)
    r.section("Packages")
    check_packages(r, REQUIRED)
    check_packages(r, OPTIONAL, required=False)
    check_data(r)
    if not r.failures:
        check_rules(r)
    check_models(r)
    check_gpu(r, a.gpu)

    print()
    if r.failures:
        print(f"{len(r.failures)} problem(s) found:")
        for f in r.failures:
            print(f"  - {f}")
        return 1
    print("Ready." if not r.warnings else f"Ready, with {len(r.warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
