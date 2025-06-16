gid_tools
├── .gitignore
├── README.md
├── STRUCTURE.md
├── checkpoints
│   └── diffusion_ckpt.pth
├── examples
│   ├── custom_tool_dataset.py
│   └── environment.py
├── gid_tools
│   ├── diffusion_model
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── data.py
│   │   ├── diffusion.py
│   │   ├── embeddings.py
│   │   ├── layers
│   │   │   ├── __init__.py
│   │   │   ├── downsample.py
│   │   │   ├── nin.py
│   │   │   └── upsample.py
│   │   ├── unet.py
│   │   └── utils.py
│   ├── envs
│   │   ├── __init__.py
│   │   └── tool_feedback.py
│   ├── reward_model
│   │   ├── __init__.py
│   │   └── linear_reward_model.py
│   └── viz
│       ├── __init__.py
│       ├── plots.py
│       └── utils.py
├── outputs
│   └── pretrained_diffusion
│       ├── sample_0.png
│       ├── sample_1.png
│       ├── sample_2.png
│       ├── sample_3.png
│       └── sample_4.png
├── requirements.txt
├── scripts
│   ├── download_model_weights.py
│   └── sample_pretrained_diffusion.py
└── setup.py

