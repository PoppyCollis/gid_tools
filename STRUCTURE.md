.
├── .gitignore
├── .vscode
│   └── settings.json
├── README.md
├── STRUCTURE.md
├── checkpoints
│   └── diffusion_ckpt.pth
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
│   │   └── feedback.py
│   ├── reward_model
│   │   ├── __init__.py
│   │   ├── linear_reward_model.py
│   │   └── reward_mlp.py
│   └── viz
│       ├── __init__.py
│       ├── plots.py
│       └── utils.py
├── old
│   ├── custom_tool_dataset.py
│   └── environment.py
├── requirements.txt
├── scripts
│   ├── download_model_weights.py
│   ├── pipeline
│   │   ├── evaluate.py
│   │   ├── feature_extraction.py
│   │   ├── generator.py
│   │   ├── run_pipeline.py
│   │   ├── samples
│   │   │   ├── sample_0.png
│   │   │   ├── sample_1.png
│   │   │   ├── sample_2.png
│   │   │   ├── sample_3.png
│   │   │   ├── sample_4.png
│   │   │   ├── sample_5.png
│   │   │   ├── sample_6.png
│   │   │   ├── sample_7.png
│   │   │   ├── sample_8.png
│   │   │   └── sample_9.png
│   │   └── train_reward_model.py
│   └── sample_pretrained_diffusion.py
└── setup.py

