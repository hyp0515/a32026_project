import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sys import path
path.append("./src/")
from model import CosmoCNN


# ============================================================
# Settings
# ============================================================

ROOT = "."
MAP_FILE = ROOT + "/data/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"
PARAM_FILE = ROOT + "/data/params_LH_IllustrisTNG.txt"

RESULT_DIR = ROOT + "/results/IllustrisTNG_Mtot_BATCHSIZE64/run_001"

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

N_SIM = 1000
N_MAPS_PER_SIM = 15

RANDOM_SEED = 42


# ============================================================
# Create output directory
# ============================================================

os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    RESULT_DIR,
    "best_model.pt"
)

NORM_PATH = os.path.join(
    RESULT_DIR,
    "normalization.npz"
)

SPLIT_PATH = os.path.join(
    RESULT_DIR,
    "split_indices.npz"
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "loss_history.npz"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device, flush=True)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True
    )


# ============================================================
# Load data
# ============================================================

print("Loading maps...", flush=True)

M_TNG = np.load(
    MAP_FILE,
    mmap_mode="r"
)


M_TNG = M_TNG.reshape(
    N_SIM,
    N_MAPS_PER_SIM,
    256,
    256
)

print(
    "Map shape:",
    M_TNG.shape,
    flush=True
)


cosmos_params = np.loadtxt(
    PARAM_FILE
)

print(
    "Parameter shape:",
    cosmos_params.shape,
    flush=True
)


# ============================================================
# Train / validation / test split
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

idx = rng.permutation(
    N_SIM
)

train_idx = idx[:800]
val_idx   = idx[800:900]
test_idx  = idx[900:1000]


print(
    "Train simulations:",
    len(train_idx),
    flush=True
)

print(
    "Validation simulations:",
    len(val_idx),
    flush=True
)

print(
    "Test simulations:",
    len(test_idx),
    flush=True
)


# Save split indices immediately
np.savez(
    SPLIT_PATH,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)


# ============================================================
# Normalize output parameters
# ============================================================

y_mean = cosmos_params[train_idx].mean(
    axis=0,
    keepdims=True
)

y_std = cosmos_params[train_idx].std(
    axis=0,
    keepdims=True
)

params_n = (
    cosmos_params - y_mean
) / y_std


# ============================================================
# Compute map normalization
#
# To avoid creating a huge temporary copy, calculate
# mean/std simulation by simulation.
# ============================================================

print(
    "Calculating map normalization...",
    flush=True
)

total_sum = 0.0
total_sq_sum = 0.0
total_count = 0

for sim_id in train_idx:

    x = np.asarray(
        M_TNG[sim_id],
        dtype=np.float64
    )

    total_sum += x.sum()
    total_sq_sum += np.square(x).sum()
    total_count += x.size


x_mean = total_sum / total_count

x_var = (
    total_sq_sum / total_count
    - x_mean**2
)

x_std = np.sqrt(x_var)


print(
    "x_mean:",
    x_mean,
    flush=True
)

print(
    "x_std:",
    x_std,
    flush=True
)


# Save normalization
np.savez(
    NORM_PATH,
    x_mean=x_mean,
    x_std=x_std,
    y_mean=y_mean,
    y_std=y_std
)


# ============================================================
# Dataset
# ============================================================

class CAMELSDataset(Dataset):

    def __init__(
        self,
        maps,
        params,
        sim_indices,
        x_mean,
        x_std
    ):

        self.maps = maps
        self.params = params
        self.sim_indices = sim_indices

        self.x_mean = x_mean
        self.x_std = x_std


    def __len__(self):

        return (
            len(self.sim_indices)
            * N_MAPS_PER_SIM
        )


    def __getitem__(self, idx):

        sim_local = (
            idx // N_MAPS_PER_SIM
        )

        map_id = (
            idx % N_MAPS_PER_SIM
        )

        sim_id = self.sim_indices[
            sim_local
        ]

        x = self.maps[
            sim_id,
            map_id
        ]

        y = self.params[
            sim_id
        ]


        # Normalize map
        x = (
            x - self.x_mean
        ) / self.x_std


        # Convert to tensor
        x = torch.as_tensor(
            np.asarray(x),
            dtype=torch.float32
        ).unsqueeze(0)

        y = torch.as_tensor(
            y,
            dtype=torch.float32
        )

        return x, y


# ============================================================
# Create datasets
# ============================================================

train_dataset = CAMELSDataset(
    M_TNG,
    params_n,
    train_idx,
    x_mean,
    x_std
)

val_dataset = CAMELSDataset(
    M_TNG,
    params_n,
    val_idx,
    x_mean,
    x_std
)

test_dataset = CAMELSDataset(
    M_TNG,
    params_n,
    test_idx,
    x_mean,
    x_std
)


print(
    "Train maps:",
    len(train_dataset),
    flush=True
)

print(
    "Validation maps:",
    len(val_dataset),
    flush=True
)

print(
    "Test maps:",
    len(test_dataset),
    flush=True
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = CosmoCNN().to(
    device
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

train_losses = []
val_losses = []


print(
    "\nStarting training...\n",
    flush=True
)


for epoch in range(
    NUM_EPOCHS
):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0


    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(X)

        loss = criterion(
            pred,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * X.size(0)
        )


    train_loss /= len(
        train_loader.dataset
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0


    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            loss = criterion(
                pred,
                y
            )

            val_loss += (
                loss.item()
                * X.size(0)
            )


    val_loss /= len(
        val_loader.dataset
    )


    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    best_mark = ""

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        best_mark = " <-- best"


    # Save history every epoch
    # so results survive even if the job stops.
    np.savez(
        HISTORY_PATH,
        train_loss=np.array(
            train_losses
        ),
        val_loss=np.array(
            val_losses
        )
    )


    print(
        f"Epoch {epoch+1:3d} "
        f"Train {train_loss:.6f} "
        f"Val {val_loss:.6f}"
        f"{best_mark}",
        flush=True
    )


# ============================================================
# Finished
# ============================================================

print(
    "\nTraining finished.",
    flush=True
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}",
    flush=True
)

print(
    f"Best model saved to: "
    f"{MODEL_PATH}",
    flush=True
)

print(
    f"Normalization saved to: "
    f"{NORM_PATH}",
    flush=True
)

print(
    f"Split indices saved to: "
    f"{SPLIT_PATH}",
    flush=True
)

print(
    f"Loss history saved to: "
    f"{HISTORY_PATH}",
    flush=True
)
