from transformers import SegformerConfig, SegformerForSemanticSegmentation
import torch.nn as nn
from torchinfo import summary
import torch

def get_segformer_model(model_name="nvidia/segformer-b4-finetuned-ade-512-512", 
                         in_channels=18, 
                         num_classes=1, 
                         image_size=128):
    config = SegformerConfig.from_pretrained(model_name)
    config.num_channels = in_channels
    config.num_labels = num_classes
    config.image_size = image_size

    model = SegformerForSemanticSegmentation(config)
    return model


if __name__ == "__main__":
    model = get_segformer_model()
    model.eval()
    summary(model, input_size=(1, 18, 128, 128))

# Multiple versions of SegFormer are available at https://huggingface.co/docs/transformers/en/model_doc/segformer