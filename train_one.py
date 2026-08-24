"""
train_one.py

Train ONE model at 100% target labels (in-domain CV + zero-shot + adaptation).
This is the architecture-comparison run. Do it once per model:

    python train_one.py vit
    python train_one.py convnext_tiny
    python train_one.py convnext_large
    python train_one.py segformer_v0
    python train_one.py segformer_v5

Results go to results/<model>/  (resume-aware: safe to kill and re-run).
Seeds (32,34,36) and 5-fold CV are applied automatically.
"""

import sys

from models import ALL_MODELS
from run_experiments import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python train_one.py <model>")
        print("models:", ALL_MODELS)
        sys.exit(1)
    model = sys.argv[1]
    if model not in ALL_MODELS:
        sys.exit(f"unknown model '{model}'. options: {ALL_MODELS}")
    main(models=[model], full_sweep=False)     # 100% budget only
