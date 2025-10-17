# Diffusion Pick-and-Place (6-DoF Robotic Manipulation)

This project implements a **diffusion-based policy** using a **U-Net architecture** to perform **6-DoF pick-and-place manipulation** within the [Ravens](https://github.com/google-research/ravens) framework.  
By conditioning a diffusion model on visual observations and goal specifications, the system learns to generate continuous pick-and-place trajectories that achieve robust manipulation performance.

## 🎥 Demo
<p align="center">
  <a href="https://imgur.com/a/yNV73eJ">
    <img src="https://i.imgur.com/H8tmnRb.gif" width="600" alt="Diffusion pick and place demo">
  </a>
</p>
---

## 🚀 Overview

This work builds upon the **Ravens** benchmark by **Google Research**, extending it with a **diffusion-based policy network** trained for end-to-end robotic manipulation.

Key modifications:
- Added a **U-Net diffusion model** for action generation.
- Integrated a **custom DiffusionAgent** class for inference and training.
- Extended the **`utils.py`** file to support sampling and denoising processes.
- Demonstrated successful **6-DoF pick-and-place** with stable convergence.

---

## 🧩 Method

1. **Training Data:**  
   Collected expert trajectories from Ravens’ pick-and-place environment.

2. **Model Architecture:**  
   - Denoising Diffusion Probabilistic Model (DDPM)  
   - U-Net backbone with time-step conditioning  
   - Predicts 6-DoF end-effector poses for grasp and place actions

3. **Training Objective:**  
   \[
   \mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \big[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \big]
   \]
   minimizing reconstruction error between true and predicted noise.

4. **Inference:**  
   At inference, the model iteratively denoises Gaussian samples into precise pick-and-place actions.

---

## 🦾 Results

The diffusion policy was successfully able to:
- Generate smooth and physically feasible pick-and-place trajectories  
- Generalize across varying object poses  
- Achieve high success rates without explicit motion planning  

🎥 **Demo Video:**  
[![Demo Video](https://imgur.com/a/yNV73eJ)](https://imgur.com/a/yNV73eJ)

---

## 🧠 Repository Structure

```bash
diffusion-pick-and-place/
├── diffusion_agent.py      # Custom diffusion policy and training loop
├── unet_model.py           # U-Net backbone for DDPM
├── utils.py                # Extended Ravens utilities for diffusion support
├── train.py                # Training entry point
├── eval.py                 # Evaluation script
└── README.md               # You are here
