# gid_tools/envs/classifier/train.py

import os
import configparser
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from tqdm import trange
from pathlib import Path

from gid_tools.envs.training_functions.classifier.cnn import ToolCNN
from gid_tools.envs.training_functions.classifier.load_dataset import load_data

def main():
    # 1) load config from the same folder as this script
    script_dir = Path(__file__).resolve().parent
    cfg = configparser.ConfigParser()
    cfg.read(script_dir / "config_cnn.ini")

    # 2) pull hyper‐params
    epochs   = int(cfg["training"]["epochs"])
    lr       = float(cfg["training"]["learning_rate"])
    opt_name = cfg["training"]["optimizer"].lower()

    # 3) resolve where to save the model
    #    respects [paths] model_output = checkpoints/model.pth
    model_rel  = cfg["paths"]["model_output"]
    output_path = (script_dir / model_rel).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Will save checkpoint to: {output_path}")
    
    # 4) data
    train_loader, _ = load_data()   

    # 5) model + device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(cfg["model"]["num_classes"])
    model = ToolCNN(num_classes=num_classes).to(device)

    # 6) optimizer
    optimizer = Adam(model.parameters(), lr=lr) if opt_name=="adam" else SGD(model.parameters(), lr=lr)

    # 7) training loop
    model.train()
    for epoch in trange(1, epochs+1, desc="Epochs"):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}")

    # 8) save
    torch.save(model.state_dict(), str(output_path))
    print(f"✓ Model saved to {output_path}")
    
if __name__ == "__main__":
    main()
