import torch
import matplotlib.pyplot as plt
import pandas as pd

# the reason for this files is to compute the average channel contributions for the entire dataset so that unnecessary channels could be removed
# this is done by computing the gradients of the output with respect to the input channels


Band_names = ["Blue","Green","Red","NIR","SWIR","Elevation/DEM","Coherence","dh/dt","Divergence","Persistant Scatter",
              "Profile Curvature","Slope Angle","Slope Aspect","Tangential Curvature","Thermal Infrared","Unsphericity","Velocity","Backscatter"]

def compute_average_channel_contributions(model, dataloader, device, band_names= Band_names):
    model.eval()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    sample_batch = next(iter(dataloader))[0]
    num_bands = sample_batch.shape[1]
    channel_sums = torch.zeros(num_bands).to(device)
    total_batches = 0

    for X, Y in dataloader:
        X = X.to(device)
        X.requires_grad = True
        model.zero_grad()

        # Forward pass
        output = model(X)
        output = output.logits  
        output = torch.sigmoid(output)  
        

        loss = output.mean()
        loss.backward()

        grads = X.grad.abs().mean(dim=[0, 2, 3])  # Mean over batch, H, W
        channel_sums += grads
        total_batches += 1

    # Normalize
    contributions = (channel_sums / total_batches)
    contributions = contributions / contributions.sum() * 100  

    contributions_np = contributions.detach().cpu().numpy()

    
    if band_names is None:
        band_names = [f"Band {i+1}" for i in range(num_bands)]

    band_data = pd.DataFrame({"Band": band_names,"Contribution": contributions_np}, index=range(1, len(band_names)+1))
    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(band_names, contributions_np, color='lightseagreen')
    plt.xticks(rotation=60)
    plt.ylabel("Contribution (%)")
    plt.title("Overall Contribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return band_data

# callable example usage:
# band_data = compute_average_channel_contributions(model, test_loader, device)