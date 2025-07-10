# === 1. Cargar y preparar los datos ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1.1 Cargar el dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)

# 1.2 Dividir en entrenamiento y test (25% test)
train, test = train_test_split(data, test_size=0.25, random_state=42)

# 1.3 Separar features y target
train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()
test_x = test.drop(["quality"], axis=1).values
test_y = test[["quality"]].values.ravel()

# 1.4 Dividir entrenamiento en train/validación (20% validación)
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

# === 2. Normalización y definición del modelo ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 2.1 Normaliza las características
mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x_norm = (train_x - mean) / std
valid_x_norm = (valid_x - mean) / std
test_x_norm = (test_x - mean) / std

# 2.2 Crea tensores y DataLoaders
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

train_ds = TensorDataset(to_tensor(train_x_norm), to_tensor(train_y))
valid_ds = TensorDataset(to_tensor(valid_x_norm), to_tensor(valid_y))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)

# 2.3 Selecciona dispositivo (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Entrenando en:", device)

# 2.4 Define la arquitectura de la red neuronal
class WineNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# === 3. Entrenamiento y seguimiento con MLflow ===
import mlflow
import mlflow.pytorch

with mlflow.start_run(run_name="modelo-basico-pytorch") as run:
    # TODO: 1. Registra los hiperparámetros en MLflow
    learning_rate = 0.01
    momentum = 0.5
    epochs = 10
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("momentum", momentum)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("architecture", "64-32-1")
    
    # Inicializa el modelo, optimizador y función de pérdida
    model = WineNet(in_dim=train_x.shape[1]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        avg_val_loss = val_loss / len(valid_loader.dataset)

        print(f"Época {epoch+1}: pérdida entrenamiento = {avg_train_loss:.4f}, pérdida validación = {avg_val_loss:.4f}")
        
        # TODO: 2. Registra las métricas de cada época
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
    
    # TODO: 3. Registra el modelo entrenado de PyTorch en MLflow
    mlflow.pytorch.log_model(model, "model")

    # TODO: 4. (Opcional) Imprime el run_id para futuras referencias o registro
    print(f"Ejecución de MLflow completada. Run ID: {run.info.run_id}")

# === FIN DEL EJERCICIO PRINCIPAL ===

# Explicación rápida:
# - Cambia los hiperparámetros y ejecuta de nuevo el código para comparar experimentos en la interfaz de MLflow.
# - Usa `mlflow ui` en la terminal y navega a http://localhost:5000 para ver y comparar los resultados.
