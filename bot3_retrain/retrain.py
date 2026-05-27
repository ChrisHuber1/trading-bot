"""GPU training pipeline for Bot3 models.

Orchestrates feature pull, training, evaluation, and deployment
of new model versions. Runs on an RTX 4070 Ti with XGBoost CUDA.

Implementation details withheld -- this is a live trading system.
"""


def retrain(force: bool = False) -> None:
    """Run the full retrain pipeline."""
    raise NotImplementedError("Retrain pipeline withheld")
