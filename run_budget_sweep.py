"""
run_budget_sweep.py

Few-shot BUDGET SWEEP for ONE model: trains adaptation at
    1%, 5%, 10%, 25%, 50%, 100%  of the target training pool.
This produces the few-shot curve (fig_fewshot_<model>.pdf).

    python run_budget_sweep.py segformer_v0
    python run_budget_sweep.py vit

Results merge into results/<model>/  (resume-aware). If you already ran
train_one.py for this model, the 100% cells are reused and only the
1/5/10/25/50% cells are computed.

Edit BUDGETS below to change the budget points.
"""

import sys

from models import ALL_MODELS
from run_experiments import main

BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python run_budget_sweep.py <model>")
        print("models:", ALL_MODELS)
        sys.exit(1)
    model = sys.argv[1]
    if model not in ALL_MODELS:
        sys.exit(f"unknown model '{model}'. options: {ALL_MODELS}")
    main(models=[model], budgets=BUDGETS)
