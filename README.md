# gid_tools
Generative inverse design of tools for robotic manipulation tasks.

Designing ad hoc tools to solve specific robotic manipulation tasks is a challenging inverse design problem. We want to be able to generate novel, modular tool designs that meet multiple objectives and generalise beyond the training distribution. For this, we propose a generative approach that leverages diffusion models to efficiently explore the tool design space. We fine-tune this generative prior via reinforcement learning (PPO).

Inspired by [SEIKO](https://github.com/zhaoyl18/SEIKO), we attempt to optimise for efficiently discovering high-reward samples with minimal feedback queries to the environment. For this, we employ a reward model with an uncertainty oracle.

Components:
- Pretrained diffusion model: trained on custom 2D (image) tool dataset. We load in the pre-trained model weights.
- Reward model: this is composed of learned features taken from the Unet middle block and put through an MLP to reduce to 16 dim feature map. We then have a linear projection of these features to a 1 dimensional scalar prediction.
- Feedback from environment: here we can plug in a generated image and return reward feedback based on any arbitrary reward function we choose to define. Later, we can plug in a simulated robotic manipulation task.
- Fine-tuning using RL: ...

