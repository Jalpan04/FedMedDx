"""
Bridge file exposing Person 1's CXR implementation to the federated core launcher.
Do not modify this file. Implement your code in modules/cxr/cxr_module.py.
"""

from modules.cxr.cxr_module import (
    get_model,
    get_hospital_partitions,
    train_one_round,
    evaluate,
    explain
)

__all__ = ["get_model", "get_hospital_partitions", "train_one_round", "evaluate", "explain"]
