import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from mlflow.models import infer_signature
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Load the wine quality dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)

# Create train/validation/test splits
train, test = train_test_split(data, test_size=0.25, random_state=42)
train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()
test_x = test.drop(["quality"], axis=1).values
test_y = test[["quality"]].values.ravel()

# Further split training data for validation
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

# Normalize inputs
mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x_norm = (train_x - mean) / std
valid_x_norm = (valid_x - mean) / std
test_x_norm = (test_x - mean) / std

# For signature inference
signature = infer_signature(train_x_norm, train_y)

# Torch datasets/loaders
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

train_ds = TensorDataset(to_tensor(train_x_norm), to_tensor(train_y))
valid_ds = TensorDataset(to_tensor(valid_x_norm), to_tensor(valid_y))
test_ds = TensorDataset(to_tensor(test_x_norm), to_tensor(test_y))


# ---- PyTorch Model ----
class WineNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # TODO: Define the net with input in_dim output 1
        self.net = nn.Sequential(
        )

    def forward(self, x):
        return self.net(x)

def rmse_loss(pred, target):
    return torch.sqrt(nn.functional.mse_loss(pred, target))

def train_torch_model(learning_rate, momentum, epochs=10, patience=3, batch_size=64, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WineNet(in_dim=train_x.shape[1]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0

    train_losses = []
    valid_losses = []
    train_rmses = []
    valid_rmses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * xb.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Compute train RMSE
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                all_preds.append(pred.cpu())
                all_targets.append(yb.cpu())
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            train_rmse = rmse_loss(all_preds, all_targets).item()
            train_rmses.append(train_rmse)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_preds.append(pred.cpu())
                val_targets.append(yb.cpu())
        avg_val_loss = val_loss / len(valid_loader.dataset)
        valid_losses.append(avg_val_loss)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_rmse = rmse_loss(val_preds, val_targets).item()
        valid_rmses.append(val_rmse)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: train loss={avg_train_loss:.4f}, val loss={avg_val_loss:.4f}, val RMSE={val_rmse:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "train_loss": train_losses,
        "val_loss": valid_losses,
        "train_rmse": train_rmses,
        "val_rmse": valid_rmses,
    }

    return {
        "model": model,
        "val_rmse": valid_rmses[-(patience+1)] if len(valid_rmses) > (patience+1) else valid_rmses[-1],
        "val_loss": valid_losses[-(patience+1)] if len(valid_losses) > (patience+1) else valid_losses[-1],
        "history": history,
        "epochs_trained": len(history["train_loss"]),
        "scaler_mean": mean,
        "scaler_std": std
    }


def objective(params):
    with mlflow.start_run(nested=True):
        
        # TODO: log params learning_rate, momentum, optimizer, architecture
        
        result = train_torch_model(
            learning_rate=params["learning_rate"],
            momentum=params["momentum"],
            epochs=15,
        )

        # TODO: log metrics val_rmse, val_loss, epochs_trained
  

        # Save PyTorch model + scaler info for inference
        class TorchModelWrapper(torch.nn.Module):
            def __init__(self, model, mean, std):
                super().__init__()
                self.model = model
                self.mean = torch.tensor(mean, dtype=torch.float32)
                self.std = torch.tensor(std, dtype=torch.float32)
            def forward(self, x):
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                x = (x - self.mean) / self.std
                return self.model(x)
        wrapped_model = TorchModelWrapper(result["model"], result["scaler_mean"], result["scaler_std"])
        
        # TODO: log your model
        

        # Log training curves as artifacts
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(result["history"]["train_loss"], label="Training Loss")
        plt.plot(result["history"]["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(result["history"]["train_rmse"], label="Training RMSE")
        plt.plot(result["history"]["val_rmse"], label="Validation RMSE")
        plt.title("Model RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curves.png")
        mlflow.log_artifact("training_curves.png")
        plt.close()
        return {"loss": result["val_rmse"], "status": STATUS_OK}

# TODO: Define search space for hyperparameters with learning_rate and momentum
search_space = {}

print("Search space defined:")
print("- Learning rate: 1e-5 to 1e-1 (log-uniform)")
print("- Momentum: 0.0 to 0.9 (uniform)")

# TODO: Create or set experiment
experiment_name = 

print(f"Starting hyperparameter optimization experiment: {experiment_name}")
print("This will run 15 trials to find optimal hyperparameters...")


# TODO: start the mlflow run login the params starting the trials and log metrics and params
# trials = Trials()
#     best_params = fmin(
#         fn=objective,
#         space=search_space,
#         algo=tpe.suggest,
#         max_evals=15,
#         trials=trials,
#         verbose=True,
#     )
# best_trial = min(trials.results, key=lambda x: x["loss"])
#    best_rmse = best_trial["loss"]

