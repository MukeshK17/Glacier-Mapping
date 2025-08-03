
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os 
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split

# # importing loss functions
# from losses.FocalDiceTversky import CombinedLoss as FocalDiceTverskyLoss
# from losses.DiceIoUSSIM import CombinedLoss as DiceIoUSSIMLoss
# from losses.DiceBCE import CombinedLoss as DiceBCELoss

from evaluate import evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)
print(device) # CUDA based code, will run on GPU if available

def train_model(model, train_loader, val_loader, criterion, epochs=500, patience=25, lr=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay =1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,  min_lr=1e-6)

    criterion = criterion()  
    train_losses = []
    validation_losses =[]
    train_accuracies = []  
    validation_accuracy =[]
    validation_iou =[]

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0   
        total = 0     


        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            if Y.ndim == 3:
                Y = Y.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(logits, size=Y.shape[2:], mode="bilinear", align_corners=False)

            # After outputs = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            # preds = (outputs > 0.5).float()           
            correct += (preds == Y).sum().item()      
            total += Y.numel()                        

            
            loss = criterion(logits, Y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc = correct / total       
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)         


        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}")

        # Validate
        model.eval()
        with torch.no_grad():
            val_metrics, val_cm = evaluate(model, val_loader)
        val_loss = val_metrics['loss']
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.6f}, IoU: {val_metrics['iou']:.6f}, Acc: {val_metrics['accuracy']:.6f}")
        validation_losses.append(val_loss)
        validation_accuracy.append(val_metrics['accuracy'])
        validation_iou.append(val_metrics['iou'])
        scheduler.step(val_loss)


        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, validation_losses, train_accuracies, validation_accuracy, validation_iou
