# gid_tools
Generative inverse design of tools for robotic manipulation tasks.

Designing ad hoc tools to solve specific robotic manipulation tasks is a challenging inverse design problem. We want to be able to generate novel, modular tool designs that meet multiple objectives and generalise beyond the training distribution. For this, we propose a generative approach that leverages diffusion models to efficiently explore the tool design space. We fine-tune this generative prior via reinforcement learning (PPO).

Inspired by [SEIKO](https://github.com/zhaoyl18/SEIKO), we attempt to optimise for efficiently discovering high-reward samples with minimal feedback queries to the environment. For this, we employ a reward model with an uncertainty oracle.
