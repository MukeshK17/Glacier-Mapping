import pandas as pd
import torch

# Device
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
from glacier_mapping.models.ResUNet import ResUNet
from glacier_mapping.models.SegFormerb4 import segformer_model

# Losses
from glacier_mapping.losses.FocalDiceTversky import CombinedLoss as FocalDiceTverskyLoss
from glacier_mapping.losses.DiceIoUSSIM import CombinedLoss as DiceIoUSSIMLoss
from glacier_mapping.losses.DiceBCE import CombinedLoss as DiceBCELoss

# Training and Evaluation
from glacier_mapping.training.train import train_model
from glacier_mapping.training.evaluate import evaluate

# Dataloader
from glacier_mapping.data.dataloader import create_dataloaders, get_test_loader

# Dataset configuration
dataset_configs = {
    "himachal": ("path1", "path2", 1.0, [5, 17, 7, 1, 4, 2]),
    "himlad": ("path3", "path4", 0.3, None),
    "sikkim": ("path5", "path6", 0.3, None),
    "kashmir": ("path7", "path8", 0.3, None),
    "uttrakhand": ("path9", "path10", 0.3, None),
}
train_loader, val_loader = create_dataloaders(dataset_configs)

# Test loaders
test_loader1 = get_test_loader("path1", "path2")
test_loader2 = get_test_loader("path3", "path4")
test_loader3 = get_test_loader("path5", "path6")
test_loader4 = get_test_loader("path7", "path8")
test_loader5 = get_test_loader("path9", "path10")

# Model + Loss
model = segformer_model().to(device) # or ResUNet()
criterion = FocalDiceTverskyLoss()

# Train

train_data = train_model(
    model,
    train_loader,
    val_loader,
    criterion = criterion,
    epochs=500,
    patience=25,
    lr=1e-5)

model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Evaluate
# Note:
# Returns (metrics, predictions, ground truths, rasters, confusion matrix)
# Metrics include: loss, IoU, accuracy, precision, recall, f1, dice, kappa

test_metrics1, pred1, gt1, rt1, cm1 = evaluate(model, test_loader1, return_predictions = True)
test_metrics2, pred2, gt2, rt2, cm2 = evaluate(model, test_loader2, return_predictions = True)
test_metrics3, pred3, gt3, rt3, cm3 = evaluate(model, test_loader3, return_predictions = True)
test_metrics4, pred4, gt4, rt4, cm4 = evaluate(model, test_loader4, return_predictions = True)
test_metrics5, pred5, gt5, rt5, cm5 = evaluate(model, test_loader5, return_predictions = True)

# Output

def get_test_output():
    return {
        "Himachal": (test_metrics1, pred1, gt1, rt1, cm1),
        "Himachal Ladakh": (test_metrics2, pred2, gt2, rt2, cm2),
        "Sikkim": (test_metrics3, pred3, gt3, rt3, cm3),
        "Kashmir": (test_metrics4, pred4, gt4, rt4, cm4),
        "Uttrakhand": (test_metrics5, pred5, gt5, rt5, cm5),
    }

# Report Metrices

Region_names = ['Himachal', 'Himachal Ladakh', 'Sikkim', 'Kashmir', 'Uttrakhand']
metrics_list = [test_metrics1, test_metrics2, test_metrics3, test_metrics4, test_metrics5]

df = pd.DataFrame(metrics_list)
df.index = Region_names

print(df)

