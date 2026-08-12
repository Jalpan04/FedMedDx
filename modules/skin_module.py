"""
Bridge file exposing Person 2's Skin Lesion implementation to the federated core launcher.
Do not modify this file. Implement your code in modules/skin/skin_module.py.
"""

from modules.skin.skin_module import (
    get_model,
    get_hospital_partitions,
    train_one_round,
    evaluate,
    explain
)

__all__ = ["get_model", "get_hospital_partitions", "train_one_round", "evaluate", "explain"]
