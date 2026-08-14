# FedMedDx Modality Interface Contract

Every disease module under `modules/` MUST implement this contract without modification to function with the FedRep federated core.

## Required Functions

### 1. `get_model() -> torch.nn.Module`
Returns a ResNet18 model initialized with pre-trained weights, modified with a custom classification head (`fc`) for the specific disease task:
- `covid_module.py`: 4 classes (`COVID-19`, `Normal`, `Lung Opacity`, `Viral Pneumonia`)
- `pneumonia_module.py`: 2 classes (`NORMAL`, `PNEUMONIA`)
- `tb_module.py`: 2 classes (`Normal`, `Tuberculosis`)
- `pneumothorax_module.py`: 2 classes (`Normal`, `Pneumothorax`)

### 2. `get_hospital_partitions(num_hospitals: int = 3, alpha: float = 0.5) -> List[Tuple[DataLoader, DataLoader]]`
Loads local disease data from `data/<modality>/` and partitions it across `num_hospitals` clients using a Dirichlet non-IID distribution ($\alpha$). Returns a list of `(train_loader, val_loader)` tuples with standardized 3-channel RGB `(3, 224, 224)` inputs.

### 3. `train_one_round(model: torch.nn.Module, train_loader: DataLoader, epochs: int, device: str) -> Tuple[dict, int, float]`
Executes local training for one federated round using FedRep alternating optimization (Phase 1: train head with frozen backbone; Phase 2: train backbone with frozen head). Returns `(model.state_dict(), num_samples_trained, average_training_loss)`.

### 4. `evaluate(model: torch.nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, Dict[str, float]]`
Evaluates the personalized model on the local hospital validation partition. Returns `(loss, {"accuracy": float, "f1_macro": float, "auc_ovr": float})`.

### 5. `explain(model: torch.nn.Module, image: torch.Tensor, device: str) -> Tuple[int, float, np.ndarray]`
Generates Grad-CAM visual explanation from `model.layer4[-1]`. Returns `(predicted_class_idx, confidence_score, overlay_rgb_array)`.
