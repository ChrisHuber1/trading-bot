"""Model inference and trade scoring.

Loads a trained XGBoost model and scores candidate trades
with a confidence value. Only trades above the confidence
threshold are forwarded to the execution layer.

Implementation details withheld -- this is a live trading system.
"""


def score_trade(pair: str, features: dict) -> float:
    """Score a candidate trade. Returns confidence 0.0-1.0."""
    raise NotImplementedError("Inference withheld")
