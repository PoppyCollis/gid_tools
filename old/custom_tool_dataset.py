import numpy as np
from torch.utils.data import Dataset
import json
import torchvision.transforms as transforms
from gid_tools.helpers.plots import make_bracket_image, render_T_image, render_V_image

class DiffusionToolDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path) as f:
            self.records = [json.loads(line) for line in f]
        self.transform = transforms.Compose([
            transforms.ToTensor(),                   # → [0,1]
            transforms.Resize((32,32)),              # → 32×32
            transforms.Lambda(lambda x: x*2 - 1)      # → [-1,1]
        ])
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        L = np.array(r['lengths']); T = np.array(r['thicknesses'])
        A = np.array(r['angles']); phi = r['phi']; fill = r['fill']
        lab = r.get('label','')
        if lab == "T":
            pil = render_T_image(L, T, A, phi)
        elif lab == "V":
            pil = render_V_image(L, T, A, phi)
        else:
            pil = make_bracket_image(L, T, A, phi, fill=fill)
        return self.transform(pil)