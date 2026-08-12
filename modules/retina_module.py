"""
Bridge file exposing Person 4's Retinal Fundus implementation to the federated core launcher.
Do not modify this file. Implement your code in modules/retina/retina_module.py.
"""

from modules.retina.retina_module import (
    get_model,
    get_hospital_partitions,
    train_one_round,
    evaluate,
    explain
)

__all__ = ["get_model", "get_hospital_partitions", "train_one_round", "evaluate", "explain"]
