import flwr as fl
import torch

class FlowerNumPyClient(fl.client.NumPyClient):
    """Generic Flower NumPyClient that wraps any contract-compliant modality module."""
    
    def __init__(self, module, train_loader, val_loader, device: str = "cpu"):
        self.module = module
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = self.module.get_model().to(self.device)

    def get_parameters(self, config):
        """Return model state dictionary weights as numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Load global model parameters into local PyTorch model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Execute one local training round for this hospital client."""
        self.set_parameters(parameters)
        epochs = config.get("local_epochs", 1)
        state_dict, num_examples, loss = self.module.train_one_round(
            self.model, self.train_loader, epochs=epochs, device=self.device
        )
        return self.get_parameters(config={}), num_examples, {"loss": float(loss)}

    def evaluate(self, parameters, config):
        """Evaluate current model parameters on local validation set."""
        self.set_parameters(parameters)
        loss, metrics = self.module.evaluate(
            self.model, self.val_loader, device=self.device
        )
        return float(loss), len(self.val_loader.dataset), metrics
