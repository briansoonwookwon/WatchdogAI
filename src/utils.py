import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split


def split_data(dataset, batch_size=32, train_size=0.8):
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("{:<10} {:<10} {:<10}".format("Train", "Val", "Test"))
    print("{:<10} {:<10} {:<10}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    print("{:<10} {:<10} {:<10}".format(len(train_loader), len(val_loader), len(test_loader)))
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