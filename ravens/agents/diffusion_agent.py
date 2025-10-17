# coding=utf-8
# Copyright 2024 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion Agent with U-Net encoder."""
import os
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"[Agent] GPU enabled: {len(physical_devices)} GPU(s)")

import random
import numpy as np
from ravens import tasks
from ravens.models.diffusion import DiffusionModel, make_encoder, make_time_table, make_schedule
from ravens.tasks import cameras
from ravens.utils import utils
import glob
import re

class DiffusionAgent:
    """Agent that uses a Diffusion Model with U-Net encoder."""

    def __init__(self, name, task, root_dir, n_rotations=36):
        self.name = name
        if isinstance(task, str):
            task = tasks.names[task]()
        
        self.task = task    
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = n_rotations
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        
        # Camera config
        try:
            self.cam_config = cameras.RealSenseD415.CONFIG
        except AttributeError:
            self.cam_config = cameras.Kinect.CONFIG
        
        self.bounds = np.array(self.task.bounds, dtype=np.float32)
        print("[Agent] task:", type(self.task).__name__)
        print("[Agent] bounds:\n", self.bounds)
        
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        self.H = self.in_shape[0]   # 320
        self.W = self.in_shape[1]   # 160
        
        # Initialize models with U-Net encoder
        print("[Agent] Creating U-Net encoder with spatial softmax...")
        self.encoder = make_encoder(self.in_shape, d=256, use_spatial_softmax=True)
        self.t_table = make_time_table(T=100, d=128)
        self.sched = make_schedule(T=100, schedule_type='cosine')
        self.eps_theta = DiffusionModel(a_dim=6, h_dim=256, t_dim=128, hidden_dim=512)
        self.opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        
        self.T = 100
        self.betas = self.sched['betas']
        self.alphas = self.sched['alphas']
        self.alphas_bar = self.sched['alphas_bar']
        
        print("[Agent] Models initialized successfully")

    def get_image(self, obs):
        """Stack color and height images."""
        cmap, hmap = utils.get_fused_heightmap(
            obs, self.cam_config, self.bounds, self.pix_size)
        img = np.concatenate((cmap,
                            hmap[..., None],
                            hmap[..., None],
                            hmap[..., None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def get_sample(self, dataset, augment=True):
        """Get a training sample from dataset."""
        (obs, act, _, _), _ = dataset.sample()
        img = self.get_image(obs)
        H, W = self.H, self.W

        # Extract GT poses
        p0_xyz, p0_xyzw = act['pose0']
        p1_xyz, p1_xyzw = act['pose1']
        p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
        p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)

        p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])

        # Data augmentation
        if augment:
            aug_img, new_pixels, _, transform_params = utils.perturb(img, [p0, p1])
            img = aug_img
            p0, p1 = new_pixels
            theta, _, _ = transform_params
            dtheta = float(theta)
            if abs(dtheta) > np.pi + 1e-6:
                dtheta = np.deg2rad(dtheta)
            p0_theta = p0_theta + dtheta
            p1_theta = p1_theta + dtheta

        # Convert to pixel indices
        r0, c0 = int(p0[0]), int(p0[1])
        r1, c1 = int(p1[0]), int(p1[1])

        # Bounds checking
        assert 0 <= c0 < W and 0 <= r0 < H, (r0, c0, H, W)
        assert 0 <= c1 < W and 0 <= r1 < H, (r1, c1, H, W)

        # Pack as [col, row, theta] for both pick and place
        a = np.array([c0, r0, p0_theta, c1, r1, p1_theta], dtype=np.float32)
        a = utils.normalize(a, W=W, H=H)

        return img, a
  
    def train(self, dataset, writer=None, augment=True):
        """Train on a dataset sample with batching."""
        
        # GET MULTIPLE SAMPLES (batch)
        batch_size = 32  # Increase from 1 to 32
        imgs, actions = [], []
        
        for _ in range(batch_size):
            img, a0 = self.get_sample(dataset, augment=augment)
            imgs.append(img)
            actions.append(a0)
        
        # Stack into batches
        with tf.device('/GPU:0'):
            img = tf.convert_to_tensor(np.stack(imgs), dtype=tf.float32)  # (32, H, W, C)
            a0 = tf.convert_to_tensor(np.stack(actions), dtype=tf.float32)  # (32, 6)
            B = tf.shape(a0)[0]

            # Sample timestep and noise
            t = tf.random.uniform((B,), 1, self.T+1, dtype=tf.int32)
            eps = tf.random.normal(tf.shape(a0), dtype=tf.float32)

            # Forward pass
            with tf.GradientTape() as tape:
                h = self.encoder(img, training=True)
                t_emb = tf.gather(self.t_table, t)

                a_bar_t = tf.gather(self.alphas_bar, t-1)
                a_bar_t = tf.reshape(a_bar_t, (-1, 1))
                a_t = tf.sqrt(a_bar_t) * a0 + tf.sqrt(1.0 - a_bar_t) * eps

                eps_hat = self.eps_theta([a_t, t_emb, h], training=True)
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(eps - eps_hat), axis=-1))

            # Update
            vars_ = (self.encoder.trainable_variables +
                    self.eps_theta.trainable_variables +
                    [self.t_table])
            grads = tape.gradient(loss, vars_)
            grads, global_norm = tf.clip_by_global_norm(grads, 1.0)
            self.opt.apply_gradients(zip(grads, vars_))

        # Logging
        step = self.total_steps + 1
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('train_loss/diffusion', loss, step=step)
                tf.summary.scalar('train/gradient_norm', global_norm, step=step)
        
        if (step % 100) == 0:
            print(f'Train Iter: {step} Loss: {loss.numpy():.4f} GradNorm: {global_norm.numpy():.4f}')

        self.total_steps = step

    def validate(self, dataset, writer=None):
        """Validation (skipped for now)."""
        print('Skipping validation.')

    def act(self, obs, info=None, goal=None):
        """Predict action using DDIM sampling."""
        tf.keras.backend.set_learning_phase(0)
        img = self.get_image(obs)[None, ...]
        H, W = self.H, self.W
        h = self.encoder(img, training=False)
        
        # Start from pure noise
        a = tf.random.normal((1, 6), mean=0.0, stddev=1.0)
        
        # DDIM sampling with 100 steps
        n_steps = 100
        idxs = tf.cast(tf.round(tf.linspace(self.T, 1, n_steps)), tf.int32)
        
        for t in tf.unstack(idxs):
            t_b = tf.reshape(t, (1,))
            t_emb = tf.gather(self.t_table, t_b)
            eps_hat = self.eps_theta([a, t_emb, h], training=False)
            
            # Get alpha values
            a_bar_t = tf.gather(self.alphas_bar, t_b - 1)
            a_bar_tm1 = tf.gather(self.alphas_bar, tf.maximum(t_b - 2, 0))
            
            a_bar_t = tf.reshape(a_bar_t, (-1, 1))
            a_bar_tm1 = tf.reshape(a_bar_tm1, (-1, 1))
            
            # Predict x_0
            sqrt_a_bar_t = tf.sqrt(a_bar_t)
            sqrt_one_minus_a_bar_t = tf.sqrt(1.0 - a_bar_t)
            
            a0_hat = (a - sqrt_one_minus_a_bar_t * eps_hat) / (sqrt_a_bar_t + 1e-8)
            a0_hat = tf.clip_by_value(a0_hat, -2.0, 2.0)
            
            # DDIM step
            if t_b > 1:
                sqrt_a_bar_tm1 = tf.sqrt(a_bar_tm1)
                sqrt_one_minus_a_bar_tm1 = tf.sqrt(1.0 - a_bar_tm1)
                dir_xt = sqrt_one_minus_a_bar_tm1 * eps_hat
                a = sqrt_a_bar_tm1 * a0_hat + dir_xt
            else:
                a = a0_hat
            
            a = tf.clip_by_value(a, -2.0, 2.0)
        
        # Denormalize
        a0 = a.numpy()[0]
        a0 = tf.clip_by_value(a0, -1.0, 1.0).numpy()
        a0 = utils.denormalize_and_clip(a0, W=W, H=H)
        
        col0, row0, theta0, col1, row1, theta1 = a0
        p0_pix = (int(row0), int(col0))
        p1_pix = (int(row1), int(col1))
        
        # Clip to bounds
        p0_pix = (max(0, min(H-1, p0_pix[0])), max(0, min(W-1, p0_pix[1])))
        p1_pix = (max(0, min(H-1, p1_pix[0])), max(0, min(W-1, p1_pix[1])))
        
        # Debug visualization
        cmap = obs['color'][0].copy()
        for (r, c) in [p0_pix, p1_pix]:
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    r2 = max(0, min(cmap.shape[0]-1, r+dr))
                    c2 = max(0, min(cmap.shape[1]-1, c+dc))
                    cmap[r2, c2] = (0, 255, 0)
        
        import imageio
        os.makedirs("debug", exist_ok=True)
        imageio.imwrite("debug/preds.png", cmap)
        print(f"Prediction: p0={p0_pix} p1={p1_pix} | θ0={theta0:.2f} θ1={theta1:.2f}")
        
        # Convert to world coordinates
        hmap = img[0, :, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -theta0))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -theta1))
        
        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
        }
  
    def save(self):
        """Save checkpoint - encoder weights are saved automatically via save_weights."""
        import json
        
        save_dir = os.path.join(self.models_dir, f"diffusion-{self.total_steps}")
        os.makedirs(save_dir, exist_ok=True)

        # Save model weights (no need to manually save SpatialSoftmax - it's part of encoder)
        self.encoder.save_weights(os.path.join(save_dir, "encoder.h5"))
        self.eps_theta.save_weights(os.path.join(save_dir, "eps_theta.h5"))

        # Save time table
        np.save(os.path.join(save_dir, "t_table.npy"), self.t_table.numpy())
        
        # Save version info
        version_info = {
            "version": "v2_unet",
            "architecture": "unet_spatial_softmax",
            "total_steps": self.total_steps,
            "encoder_type": "UNetSpatialEncoder",
            "schedule": "cosine",
        }
        with open(os.path.join(save_dir, "model_version.json"), 'w') as f:
            json.dump(version_info, f, indent=2)
        
        print(f"✅ Saved diffusion model at {save_dir}")

    def load(self, n_iter):
        """Load checkpoint at specific iteration."""
        import json
        
        # Build models once to create variables
        print("[Agent] Building models for loading...")
        _ = self.encoder(tf.zeros([1, *self.in_shape], dtype=tf.float32))
        _ = self.eps_theta([
            tf.zeros([1, 6], dtype=tf.float32),
            tf.zeros([1, 128], dtype=tf.float32),
            tf.zeros([1, 256], dtype=tf.float32),
        ])

        # Load weights
        ckpt_dir = os.path.join(self.models_dir, f"diffusion-{n_iter}")
        
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
        
        # Check version
        version_file = os.path.join(ckpt_dir, "model_version.json")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_info = json.load(f)
            print(f"[Agent] Loading checkpoint version: {version_info.get('version', 'unknown')}")
            print(f"[Agent] Architecture: {version_info.get('architecture', 'unknown')}")
        
        # Load weights (SpatialSoftmax is automatically loaded as part of encoder)
        self.encoder.load_weights(os.path.join(ckpt_dir, "encoder.h5"))
        self.eps_theta.load_weights(os.path.join(ckpt_dir, "eps_theta.h5"))
        
        # Load time table
        tpath = os.path.join(ckpt_dir, "t_table.npy")
        if os.path.exists(tpath):
            self.t_table.assign(np.load(tpath))
        
        self.total_steps = n_iter
        print(f"✅ Loaded checkpoint: {ckpt_dir}")

    def load_latest(self):
        """Load the most recent checkpoint."""
        import json
        
        # Build models once
        print("[Agent] Building models for loading...")
        _ = self.encoder(tf.zeros([1, *self.in_shape], tf.float32))
        _ = self.eps_theta([tf.zeros([1, 6]), tf.zeros([1, 128]), tf.zeros([1, 256])])

        # Find latest checkpoint
        pat = os.path.join(self.models_dir, "diffusion-*")
        cands = sorted(glob.glob(pat), key=lambda p: int(re.findall(r"diffusion-(\d+)$", p)[0]))
        
        if not cands:
            raise FileNotFoundError(f"No checkpoints found in {self.models_dir}")
        
        latest = cands[-1]
        
        # Check version
        version_file = os.path.join(latest, "model_version.json")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_info = json.load(f)
            print(f"[Agent] Loading checkpoint version: {version_info.get('version', 'unknown')}")
            arch = version_info.get('architecture', 'unknown')
            print(f"[Agent] Architecture: {arch}")
            
            # Warn if architecture mismatch
            if 'unet' not in arch.lower():
                print("⚠️  WARNING: Checkpoint may not be compatible with U-Net architecture!")
        
        # Load weights (SpatialSoftmax loaded automatically with encoder)
        try:
            self.encoder.load_weights(os.path.join(latest, "encoder.h5"))
            self.eps_theta.load_weights(os.path.join(latest, "eps_theta.h5"))
            
            t_path = os.path.join(latest, "t_table.npy")
            if os.path.exists(t_path):
                self.t_table.assign(np.load(t_path))
            
            # Extract iteration number
            iteration = int(re.findall(r"diffusion-(\d+)$", latest)[0])
            self.total_steps = iteration
            
            print(f"✅ Loaded latest checkpoint: {latest}")
            
        except ValueError as e:
            if "layers" in str(e):
                print(f"\n❌ ERROR: Model architecture mismatch!")
                print(f"The checkpoint was saved with a different architecture.")
                print(f"Checkpoint: {latest}")
                print(f"\nOptions:")
                print(f"1. Delete old checkpoints and retrain: rm -rf {self.models_dir}/diffusion-*")
                print(f"2. Use a different checkpoint directory")
                print(f"3. Revert to the old architecture\n")
            raise