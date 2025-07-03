#!/usr/bin/env python
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from custom_tool_dataset import CNNToolDataset

def main():
    # 1) locate paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.normpath(
        os.path.join(script_dir, '..', 'datasets', 'tools_dataset_classes_reduced.jsonl')
    )
    output_dir = os.path.join(script_dir, 'ground_truth_samples')
    os.makedirs(output_dir, exist_ok=True)

    # 2) load dataset
    dataset = CNNToolDataset(jsonl_path)

    # 3) group by label
    label_to_indices = defaultdict(list)
    for idx, rec in enumerate(dataset.records):
        label_to_indices[rec.get('label', '')].append(idx)

    # 4) pick up to 4 random samples per label
    samples_per_label = {
        lab: random.sample(idxs, min(4, len(idxs)))
        for lab, idxs in label_to_indices.items()
    }

    # 5) for each label, build a figure and save it
    for i, (lab, idxs) in enumerate(samples_per_label.items()):
        # one figure per tool type
        fig = plt.figure(constrained_layout=True, figsize=(16, 4))
        # when rows*cols == 1, subfigures returns the SubFigure itself
        subfig = fig.subfigures(1, 1)

        # title the subfigure
        subfig.suptitle(f"Tool type: '{lab}'", fontsize=14, y=1.02)

        # create 4 axes in a row
        axes = subfig.subplots(1, 4)
        for ax, idx in zip(axes, idxs):
            img_tensor, _ = dataset[idx]
            img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img = (img + 1) / 2  # back to [0,1] range
            ax.imshow(img.squeeze(), cmap='gray')
            ax.axis('off')

        # save and clean up
        out_path = os.path.join(output_dir, f'fig_{i}.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
