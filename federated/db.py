import sqlite3
import os
import datetime
import uuid
from typing import Dict, List, Optional

DB_PATH = os.path.join("results", "fedmeddx_experiments.db")

def init_db(db_path: str = DB_PATH):
    """Initialize SQLite database tables for experiment metric tracking."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. experiment_runs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiment_runs (
        run_id TEXT PRIMARY KEY,
        modality TEXT NOT NULL,
        strategy TEXT NOT NULL,
        alpha REAL NOT NULL,
        num_hospitals INTEGER NOT NULL,
        rounds INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. round_metrics table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS round_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        round_num INTEGER NOT NULL,
        hospital_id INTEGER NOT NULL,
        loss REAL NOT NULL,
        accuracy REAL NOT NULL,
        f1_macro REAL NOT NULL,
        auc_ovr REAL NOT NULL,
        FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
    )
    """)
    
    # 3. hospital_distributions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hospital_distributions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        hospital_id INTEGER NOT NULL,
        class_idx INTEGER NOT NULL,
        sample_count INTEGER NOT NULL,
        FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
    )
    """)
    
    conn.commit()
    conn.close()

def log_experiment_run(modality: str, strategy: str, alpha: float, num_hospitals: int, rounds: int, db_path: str = DB_PATH) -> str:
    """Log new experiment run and return generated run_id."""
    init_db(db_path)
    run_id = f"{modality}_{strategy}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO experiment_runs (run_id, modality, strategy, alpha, num_hospitals, rounds) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, modality, strategy, alpha, num_hospitals, rounds)
    )
    conn.commit()
    conn.close()
    return run_id

def log_round_metrics(run_id: str, round_num: int, hospital_id: int, loss: float, accuracy: float, f1_macro: float, auc_ovr: float, db_path: str = DB_PATH):
    """Log metric entry for a round."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO round_metrics (run_id, round_num, hospital_id, loss, accuracy, f1_macro, auc_ovr) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, round_num, hospital_id, loss, accuracy, f1_macro, auc_ovr)
    )
    conn.commit()
    conn.close()

def log_hospital_distribution(run_id: str, hospital_id: int, class_idx: int, sample_count: int, db_path: str = DB_PATH):
    """Log sample count for a specific class on a hospital client."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO hospital_distributions (run_id, hospital_id, class_idx, sample_count) VALUES (?, ?, ?, ?)",
        (run_id, hospital_id, class_idx, sample_count)
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Initializing SQLite database...")
    init_db()
    test_run = log_experiment_run("dummy", "FedAvg", 0.5, 3, 5)
    log_round_metrics(test_run, round_num=1, hospital_id=-1, loss=0.5, accuracy=0.85, f1_macro=0.82, auc_ovr=0.88)
    print(f"Database initialized and test metric logged cleanly for run: {test_run}")
