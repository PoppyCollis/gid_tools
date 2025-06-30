# gid_tools/envs/classifier/test.py

import configparser
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

from gid_tools.envs.training_functions.classifier.cnn import ToolCNN
from gid_tools.envs.training_functions.classifier.load_dataset import load_data


def main():
    # config
    cfg = configparser.ConfigParser()
    script_dir = Path(__file__).resolve().parent
    cfg.read(script_dir / "config_cnn.ini")

    # resolve model path relative to this folder
    model_rel  = cfg["paths"]["model_output"]      # e.g. "checkpoints/model.pth"
    model_path = (script_dir / model_rel).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Couldn’t find model at {model_path}. Did you train first?")
 

    # data
    _, test_loader = load_data()

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(cfg["model"]["num_classes"])
    model = ToolCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    print("Test Accuracy:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

if __name__ == "__main__":
    main()
