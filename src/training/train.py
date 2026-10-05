"""Explicit experiment policy: pretrained inference only, with no fit on test data."""
from src.evaluation.cross_validation import validation_split


def training_partition(first_test_origin: int, validation_fraction: float = 0.2):
    """Provide chronological train/validation slices for a future training extension.

    This benchmark does not optimize weights or hyperparameters. Callers must
    slice the dataset at first_test_origin before fitting any future estimator.
    """
    return validation_split(first_test_origin, validation_fraction)
