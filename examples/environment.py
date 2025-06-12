import torch
import matplotlib.pyplot as plt
from custom_tool_dataset import *
from plots import display_samples_with_feedback

def pixel_area_tensor(img: torch.Tensor, threshold: float = 0.0) -> int:
    """
    Compute the 2D area (number of 'on' pixels) of a Torch image in [-1,1].
    Assumes img.shape = [1,H,W] or [H,W].
    Counts pixels > threshold.
    """
    # squeeze channel if present
    if img.dim() == 3:
        img = img.squeeze(0)
    mask = img > threshold
    return int(mask.sum().item())

def tool_length():
    pass

def abstract_properties():
    pass

def main():

    # load in custom tool dataset
    jsonl_path = 'data/tools_dataset_classes_reduced.jsonl'
    dataset    = DiffusionToolDataset(jsonl_path)

    # reproducible random indices
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(dataset), generator=g)
    idx  = perm[:20]

    # grab image tensors and labels from the raw records
    sample_images = [dataset[i] for i in idx] # returns tensor in [-1,1]
    sample_labels = [dataset.records[i].get('label','') for i in idx]

    # compute areas and visualise
    areas = [pixel_area_tensor(img, threshold=0.0) for img in sample_images]

    display_samples_with_feedback(sample_images, sample_labels, areas)

if __name__ == "__main__":
    main()
