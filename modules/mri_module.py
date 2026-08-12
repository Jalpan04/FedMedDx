"""
Bridge file exposing Person 3's Brain MRI implementation to the federated core launcher.
Do not modify this file. Implement your code in modules/mri/mri_module.py.
"""

from modules.mri.mri_module import (
    get_model,
    get_hospital_partitions,
    train_one_round,
    evaluate,
    explain
)

__all__ = ["get_model", "get_hospital_partitions", "train_one_round", "evaluate", "explain"]
