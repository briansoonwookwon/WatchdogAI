import json
import torch
from tabulate import tabulate
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os

from src.dataset import PosterDataset


def split_data(root_dir, batch_size=32, workers=4):
    train_dataset = PosterDataset(os.path.join(root_dir, "train"), train=True)
    val_dataset = PosterDataset(os.path.join(root_dir, "val"), train=False)
    test_dataset = PosterDataset(os.path.join(root_dir, "test"), train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    headers = ["", "Train", "Val", "Test", "Total"]
    table = [
        ["Samples", len(train_dataset), len(val_dataset), len(test_dataset), 
         len(train_dataset) + len(val_dataset) + len(test_dataset)],
        ["Batches", len(train_loader), len(val_loader), len(test_loader),
         len(train_loader) + len(val_loader) + len(test_loader)]
    ]
    print(tabulate(table, headers=headers))
    return train_loader, val_loader, test_loader

def plot_history(loss_train, loss_val, acc_train, acc_val):
    """
    Plots training and validation loss and accuracy curves.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Loss plot
    ax[0].plot(loss_train, label="Train Loss")
    ax[0].plot(loss_val, label="Val Loss")
    ax[0].set_title("Loss over Epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    # Accuracy plot
    ax[1].plot(acc_train, label="Train Accuracy")
    ax[1].plot(acc_val, label="Val Accuracy")
    ax[1].set_title("Accuracy over Epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()

def print_history(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)
    headers = ["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc"]
    table = []
    for i, (tl, vl, ta, va) in enumerate(zip(
            history["train_loss"],
            history["val_loss"],
            history["train_acc"],
            history["val_acc"]), start=1):
        mark = "*" if i == history["best_epoch"] else ""
        table.append([
            f"{i}{mark}",
            f"{tl:.4f}",
            f"{vl:.4f}",
            f"{ta:.4f}",
            f"{va:.4f}"
        ])

    print(tabulate(table, headers=headers))
    print(f"\nBest val acc: {history['best_val_acc']:.4f} (Epoch {history['best_epoch']})")
    print(f"Saved model: {history['model_path']}")