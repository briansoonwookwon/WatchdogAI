import os, json
import torch
from tqdm import tqdm
from IPython.display import clear_output

from src.utils import plot_history

device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_one_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(data_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # Adjust labels to shape [batch, 1] as required by BCEWithLogitsLoss
        labels = labels.float().unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5)
        total_correct += torch.sum(preds == labels.byte()).item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5)
            total_correct += torch.sum(preds == labels.byte()).item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience=5):
    model.to(device)

    # determine model type and base save directory
    model_type = model.__class__.__name__
    save_dir = os.path.join("models", model_type)

    best_val_acc = 0.0
    best_epoch = 0
    loss_history_train, loss_history_val = [], []
    acc_history_train, acc_history_val = [], []
    epochs_no_improve = 0

    run_num = 1
    run_folder = os.path.join(save_dir, f"run_{run_num}")
    while os.path.exists(run_folder):
        run_num += 1
        run_folder = os.path.join(save_dir, f"run_{run_num}")
    os.makedirs(run_folder, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        loss_history_train.append(train_loss)
        loss_history_val.append(val_loss)
        acc_history_train.append(train_acc)
        acc_history_val.append(val_acc)

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_folder, "best_model.pth"))
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        clear_output(wait=True)
        plot_history(loss_history_train, loss_history_val, acc_history_train, acc_history_val)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch+1}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    history = {
        "train_loss": loss_history_train,
        "val_loss":   loss_history_val,
        "train_acc":  acc_history_train,
        "val_acc":    acc_history_val,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch + 1,
        "model_path":   os.path.join(run_folder, "best_model.pth")
    }
    history_path = os.path.join(run_folder, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {history_path}")
    return history

def test(model, test_loader, criterion):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5)
            total_correct += torch.sum(preds == labels.byte()).item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    # print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy