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

"""Diffusion module with U-Net encoder."""

import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import layers

# ---------- Spatial Softmax Layer ----------

class SpatialSoftmax(layers.Layer):
    """Spatial softmax layer for extracting 2D keypoints from feature maps."""
    
    def __init__(self, name='spatial_softmax'):
        super().__init__(name=name)
        self.height = None
        self.width = None
        self.pos_x = None
        self.pos_y = None
    
    def build(self, input_shape):
        # input_shape: (batch, height, width, channels)
        self.height = input_shape[1]
        self.width = input_shape[2]
        
        # Create coordinate grids
        pos_x = tf.range(self.width, dtype=tf.float32)
        pos_y = tf.range(self.height, dtype=tf.float32)
        
        pos_x = (pos_x / (self.width - 1)) * 2 - 1   # Normalize to [-1, 1]
        pos_y = (pos_y / (self.height - 1)) * 2 - 1
        
        pos_x = tf.reshape(pos_x, [1, 1, self.width, 1])
        pos_y = tf.reshape(pos_y, [1, self.height, 1, 1])
        
        self.pos_x = pos_x
        self.pos_y = pos_y
        super().build(input_shape)
    
    def call(self, features):
        # features: (B, H, W, C)
        B = tf.shape(features)[0]
        C = features.shape[-1]
        
        # Reshape to (B, H*W, C)
        features_flat = tf.reshape(features, [B, self.height * self.width, C])
        
        # Softmax over spatial dimensions for each channel
        softmax_attention = tf.nn.softmax(features_flat, axis=1)  # (B, H*W, C)
        softmax_attention = tf.reshape(softmax_attention, 
                                      [B, self.height, self.width, C])
        
        # Compute expected coordinates
        expected_x = tf.reduce_sum(softmax_attention * self.pos_x, axis=[1, 2])  # (B, C)
        expected_y = tf.reduce_sum(softmax_attention * self.pos_y, axis=[1, 2])  # (B, C)
        
        # Concatenate x and y coordinates: (B, 2*C)
        keypoints = tf.concat([expected_x, expected_y], axis=1)
        
        return keypoints


# ---------- U-Net Encoder ----------

class UNetSpatialEncoder(tf.keras.Model):
    """U-Net with spatial softmax for extracting spatial features."""
    
    def __init__(self, in_shape=(320, 160, 6), num_keypoints=32, output_dim=256):
        super().__init__(name='unet_spatial_encoder')
        self.in_shape = in_shape
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim
        
        # U-Net encoder
        self.enc1 = self._conv_block(32, name='enc1')
        self.enc2 = self._conv_block(64, name='enc2')
        self.enc3 = self._conv_block(128, name='enc3')
        self.enc4 = self._conv_block(256, name='enc4')
        self.pool = layers.MaxPooling2D(pool_size=2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, name='bottleneck')
        
        # U-Net decoder
        self.upconv4 = layers.Conv2DTranspose(256, 2, strides=2, padding='same', name='upconv4')
        self.dec4 = self._conv_block(256, name='dec4')
        
        self.upconv3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same', name='upconv3')
        self.dec3 = self._conv_block(128, name='dec3')
        
        self.upconv2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same', name='upconv2')
        self.dec2 = self._conv_block(64, name='dec2')
        
        self.upconv1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same', name='upconv1')
        self.dec1 = self._conv_block(32, name='dec1')
        
        # Convert to keypoint features
        self.feature_conv = layers.Conv2D(num_keypoints, 1, activation='relu', name='feature_conv')
        self.spatial_softmax = SpatialSoftmax()
        
        # Project to output dimension
        self.fc = layers.Dense(output_dim, activation='relu', name='fc')
        
    def _conv_block(self, filters, name):
        """Convolutional block with batch norm."""
        return tf.keras.Sequential([
            layers.Conv2D(filters, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
        ], name=name)
    
    def call(self, x, training=False):
        # Encoder
        e1 = self.enc1(x, training=training)           # (B, H, W, 32)
        e2 = self.enc2(self.pool(e1), training=training)     # (B, H/2, W/2, 64)
        e3 = self.enc3(self.pool(e2), training=training)     # (B, H/4, W/4, 128)
        e4 = self.enc4(self.pool(e3), training=training)     # (B, H/8, W/8, 256)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4), training=training)  # (B, H/16, W/16, 512)
        
        # Decoder with skip connections
        d4 = self.upconv4(b)                           # (B, H/8, W/8, 256)
        d4 = tf.concat([d4, e4], axis=-1)              # Skip connection
        d4 = self.dec4(d4, training=training)
        
        d3 = self.upconv3(d4)                          # (B, H/4, W/4, 128)
        d3 = tf.concat([d3, e3], axis=-1)
        d3 = self.dec3(d3, training=training)
        
        d2 = self.upconv2(d3)                          # (B, H/2, W/2, 64)
        d2 = tf.concat([d2, e2], axis=-1)
        d2 = self.dec2(d2, training=training)
        
        d1 = self.upconv1(d2)                          # (B, H, W, 32)
        d1 = tf.concat([d1, e1], axis=-1)
        d1 = self.dec1(d1, training=training)          # (B, H, W, 32)
        
        # Extract spatial keypoints
        features = self.feature_conv(d1)               # (B, H, W, num_keypoints)
        keypoints = self.spatial_softmax(features)     # (B, 2*num_keypoints)
        
        # Project to output dimension
        output = self.fc(keypoints)                    # (B, output_dim)
        
        return output


# ---------- Schedule & Time Embeddings ----------

def make_encoder(in_shape=(320, 160, 6), d=256, use_spatial_softmax=True):
    """Create encoder - U-Net with spatial softmax (recommended for pick-place)."""
    if use_spatial_softmax:
        return UNetSpatialEncoder(in_shape, num_keypoints=32, output_dim=d)
    else:
        # Fallback to ResNet if needed
        from ravens.models.resnet import ResNet36_4s
        x, y = ResNet36_4s(in_shape, d, prefix='enc_')
        y = layers.GlobalAveragePooling2D()(y)
        y = layers.Dense(d, activation='relu')(y)
        return tf.keras.Model(x, y, name='encoder')


def make_time_table(T=100, d=128):
    """Learnable time embedding table."""
    return tf.Variable(tf.random.normal([T+1, d], stddev=0.02), name='t_emb', trainable=True)


def make_schedule(T=100, schedule_type='cosine'):
    """Create noise schedule."""
    
    if schedule_type == 'linear':
        beta_start, beta_end = 1e-4, 0.02
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
        
    elif schedule_type == 'cosine':
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        s = 0.008
        steps = T + 1
        x = np.linspace(0, T, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0.0001, 0.9999)
        
    elif schedule_type == 'quadratic':
        beta_start, beta_end = 1e-4, 0.02
        betas = np.linspace(beta_start**0.5, beta_end**0.5, T, dtype=np.float32) ** 2
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas, axis=0)

    return {
        "betas": tf.constant(betas, dtype=tf.float32),
        "alphas": tf.constant(alphas, dtype=tf.float32),
        "alphas_bar": tf.constant(alphas_bar, dtype=tf.float32),
    }


# ---------- Epsilon-Theta Network ----------

class DiffusionModel(tf.keras.Model):
    """Improved epsilon-theta network with proper conditioning."""
    
    def __init__(self, a_dim=6, h_dim=256, t_dim=128, hidden_dim=512):
        super().__init__(name='eps_theta')
        
        # Separate projections for each input
        self.action_proj = layers.Dense(hidden_dim, activation='relu', name='action_proj')
        self.time_proj = layers.Dense(hidden_dim, activation='relu', name='time_proj')
        self.image_proj = layers.Dense(hidden_dim, activation='relu', name='image_proj')
        
        # Main processing blocks
        self.block1_dense1 = layers.Dense(hidden_dim, activation='relu', name='block1_dense1')
        self.block1_ln1 = layers.LayerNormalization(name='block1_ln1')
        self.block1_dense2 = layers.Dense(hidden_dim, name='block1_dense2')
        self.block1_ln2 = layers.LayerNormalization(name='block1_ln2')
        
        self.block2_dense1 = layers.Dense(hidden_dim, activation='relu', name='block2_dense1')
        self.block2_ln1 = layers.LayerNormalization(name='block2_ln1')
        self.block2_dense2 = layers.Dense(hidden_dim, name='block2_dense2')
        self.block2_ln2 = layers.LayerNormalization(name='block2_ln2')
        
        self.block3_dense1 = layers.Dense(hidden_dim, activation='relu', name='block3_dense1')
        self.block3_ln1 = layers.LayerNormalization(name='block3_ln1')
        self.block3_dense2 = layers.Dense(hidden_dim, name='block3_dense2')
        self.block3_ln2 = layers.LayerNormalization(name='block3_ln2')
        
        # Output
        self.out = layers.Dense(a_dim, name='out')
        
    def _apply_block(self, x, dense1, ln1, dense2, ln2, training=False):
        """Apply residual block."""
        residual = x
        x = dense1(x)
        x = ln1(x, training=training)
        x = dense2(x)
        x = ln2(x, training=training)
        return tf.nn.relu(x + residual)

    def call(self, inputs, training=False):
        a_t, t_emb, h = inputs  # (B,6), (B,128), (B,256)
        
        # Project each input
        a_feat = self.action_proj(a_t)
        t_feat = self.time_proj(t_emb)
        h_feat = self.image_proj(h)
        
        # Combine with addition (FiLM-style conditioning)
        z = a_feat + t_feat + h_feat
        
        # Process through residual blocks
        z = self._apply_block(z, self.block1_dense1, self.block1_ln1, 
                            self.block1_dense2, self.block1_ln2, training)
        z = self._apply_block(z, self.block2_dense1, self.block2_ln1,
                            self.block2_dense2, self.block2_ln2, training)
        z = self._apply_block(z, self.block3_dense1, self.block3_ln1,
                            self.block3_dense2, self.block3_ln2, training)
        
        return self.out(z)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.save_weights(path)

    def load(self, path):
        self.load_weights(path)