import os
import gdown

FILE_ID = "1-MTVljcj5gBLhVdq_ShcrFq9zI_8X4YA"
DEST    = os.path.join("checkpoints", "diffusion_ckpt.pth")
os.makedirs(os.path.dirname(DEST), exist_ok=True)

url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
gdown.download(url, DEST, quiet=False)

