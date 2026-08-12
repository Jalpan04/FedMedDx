# FedMedDx Developer Environment & Setup Guide

This guide details environment setup for all 5 team members, covering Python virtual environments, PyTorch GPU acceleration, and Kaggle API configuration.

---

## 1. Prerequisites

- Python 3.10 or higher
- NVIDIA CUDA Toolkit 11.8 / 12.1 (recommended for GPU acceleration)
- Git CLI
- Kaggle Account

---

## 2. Virtual Environment Setup

It is strongly recommended to use a dedicated virtual environment for this project.

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. GPU PyTorch Installation

To maximize training speed during local training rounds, ensure PyTorch detects your GPU:

```bash
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU availability in Python:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

---

## 4. Full Dependencies Installation

After installing PyTorch with CUDA support, install all project requirements:

```bash
pip install -r requirements.txt
```

---

## 5. Kaggle API Configuration

Every team member requires access to Kaggle for automated dataset retrieval:

1. Log into Kaggle and navigate to `https://www.kaggle.com/settings`.
2. Scroll to the **API** section and click **Create New Token**.
3. A `kaggle.json` file will download to your machine.
4. Place `kaggle.json` in the appropriate directory:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/macOS**: `~/.kaggle/kaggle.json`
5. Set appropriate permissions (Linux/macOS):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
6. Test API authentication:
   ```bash
   kaggle datasets list
   ```
