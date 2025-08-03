import torch
from torch import nn
import pandas as pd

# Device configuration
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#importing models
from models.ResUNet import ResUNet
from models.SegFormerb4 import segformer_model

#importing loss functions
from losses.FocalDiceTversky import CombinedLoss as FocalDiceTverskyLoss
from losses.DiceIoUSSIM import CombinedLoss as DiceIoUSSIMLoss
from losses.DiceBCE import CombinedLoss as DiceBCELoss

# importing training and evaluation functions
from train import train_model
from evaluate import evaluate

#importing dataloader
from dataloader import get_dataloaders, get_test_loaders

train_loader, val_loader = get_dataloaders()
test_loader1, test_loader2, test_loader3, test_loader4, test_loader5 = get_test_loaders()

model = segformer_model().to(device) # Change to ResUNet() if you want to use ResUNet model
criterion = FocalDiceTverskyLoss()  # Change to DiceIoUSSIMLoss or DiceBCELoss as needed

# Train the model
train_data = train_model(model, train_loader, val_loader, criterion = criterion, epochs=500, patience=25, lr=1e-5)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
# Evaluate the model on test datasets
#Note: here test matrices, predictions, ground truths, rasters and confusion matrices are returned in order
# test matrices contains loss, IoU, accuracy, precision, recall, f1-score, dice-score, kappa

test_metrics1, pred1, gt1, rt1, cm1 = evaluate(model, test_loader1, return_predictions = True)
test_metrics2, pred2, gt2, rt2, cm2 = evaluate(model, test_loader2, return_predictions = True)
test_metrics3, pred3, gt3, rt3, cm3 = evaluate(model, test_loader3, return_predictions = True)
test_metrics4, pred4, gt4, rt4, cm4 = evaluate(model, test_loader4, return_predictions = True)
test_metrics5, pred5, gt5, rt5, cm5 = evaluate(model, test_loader5, return_predictions = True)


# Report Metrices 
Region_names = ['Himachal', 'Himachal Ladakh', 'Sikkim', 'Kashmir', 'Uttrakhand']
metrics_list = [test_metrics1, test_metrics2, test_metrics3, test_metrics4, test_metrics5]
df = pd.DataFrame(metrics_list)
df.index = Region_names
print(df)

