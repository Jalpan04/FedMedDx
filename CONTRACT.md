# FedMedDx Modality Interface Contract (CONTRACT.md)

This contract defines the mandatory interface for all medical modality modules in `modules/`:
- `cxr_module.py` (Chest X-ray)
- `skin_module.py` (Skin Lesion)
- `mri_module.py` (Brain MRI)
- `retina_module.py` (Retinal Fundus)

Every module **MUST** expose exactly these five functions with identical signatures and return types.

---

## 1. get_model()

```python
def get_model() -> torch.nn.Module:
    """Instantiate a fresh ResNet18 model with custom classification head.
    
    Returns:
        torch.nn.Module: PyTorch ResNet18 initialized with ImageNet weights
                         and fc layer matching this modality's class count.
    """
```

---

## 2. get_hospital_partitions()

```python
def get_hospital_partitions(
    num_hospitals: int, 
    alpha: float
) -> List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]:
    """Partition modality dataset into Non-IID hospital splits using Dirichlet distribution.
    
    Args:
        num_hospitals (int): Number of simulated hospital clients.
        alpha (float): Dirichlet concentration parameter (e.g. 0.5 default, 0.1 severe non-IID).
        
    Returns:
        List[Tuple[DataLoader, DataLoader]]: List of (train_loader, val_loader) pairs,
                                             one per hospital partition.
    """
```

---

## 3. train_one_round()

```python
def train_one_round(
    model: torch.nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    epochs: int, 
    device: str
) -> Tuple[dict, int, float]:
    """Perform local training on one hospital partition for one federated round.
    
    Args:
        model (torch.nn.Module): PyTorch model instance.
        train_loader (DataLoader): Local training data loader for this hospital.
        epochs (int): Number of local training epochs per round (default 1-2).
        device (str): Device string ("cuda" if available else "cpu").
        
    Returns:
        Tuple[dict, int, float]:
            - updated_state_dict: model.state_dict() after local updates.
            - num_examples: total number of training samples processed.
            - avg_loss: average scalar training loss over all local epochs.
    """
```

---

## 4. evaluate()

```python
def evaluate(
    model: torch.nn.Module, 
    val_loader: torch.utils.data.DataLoader, 
    device: str
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model performance on local validation data.
    
    Args:
        model (torch.nn.Module): PyTorch model instance.
        val_loader (DataLoader): Local validation data loader.
        device (str): Device string ("cuda" if available else "cpu").
        
    Returns:
        Tuple[float, Dict[str, float]]:
            - loss: average validation loss.
            - metrics: dictionary containing:
                - "accuracy": float (0 to 1)
                - "f1_macro": float (0 to 1)
                - "auc_ovr": float (0 to 1)
    """
```

---

## 5. explain()

```python
def explain(
    model: torch.nn.Module, 
    image: torch.Tensor, 
    device: str
) -> Tuple[int, float, np.ndarray]:
    """Generate Grad-CAM activation heatmap overlay for model prediction.
    
    Args:
        model (torch.nn.Module): PyTorch model instance.
        image (torch.Tensor): Single image tensor of shape (3, 224, 224).
        device (str): Device string ("cuda" if available else "cpu").
        
    Returns:
        Tuple[int, float, np.ndarray]:
            - predicted_class_idx: int prediction class index.
            - confidence_score: float confidence probability (0.0 to 1.0).
            - gradcam_overlay: RGB uint8 numpy array of shape (224, 224, 3) ready for rendering.
    """
```
