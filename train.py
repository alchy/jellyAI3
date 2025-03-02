import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import os
from settings import LEARNING_RATE, WEIGHT_DECAY, PATIENCE, SCHEDULER_PATIENCE, SCHEDULER_FACTOR, WEIGHTS_DIR

def train_model(model, train_loader, val_loader, epochs, device):
    """
    Trénuje jazykový model s validační smyčkou, early stopping a schedulerem.
    
    Argumenty:
        model: LSTM model, který se má trénovat.
        train_loader: DataLoader pro trénovací data.
        val_loader: DataLoader pro validační data.
        epochs (int): Maximální počet epoch.
        device: Zařízení (CPU nebo GPU).
    
    Návratová hodnota:
        float: Nejlepší validační ztráta.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)
    model.to(device)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Vytvoření složky pro ukládání vah, pokud neexistuje
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for epoch in range(epochs):
        # Trénovací smyčka
        model.train()
        total_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validační smyčka
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Uložení modelu, pokud je validační ztráta lepší
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Uložení modelu do složky saved_weights
            torch.save(model.state_dict(), f'{WEIGHTS_DIR}/best_model.pth')
            print(f"Nejlepší model uložen do '{WEIGHTS_DIR}/best_model.pth'")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Načtení nejlepšího modelu na konci trénování
    model.load_state_dict(torch.load(f'{WEIGHTS_DIR}/best_model.pth'))
    print(f"Načten nejlepší model z '{WEIGHTS_DIR}/best_model.pth'")
    return best_val_loss
