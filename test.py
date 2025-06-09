import os

import numpy as np
import tensorflow as tf

from detector import WatermarkResNet
from embedder import AudioWatermarkEmbedder

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'           # Turn off OneDNN logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'           # Force CPU usage to avoid GPU errors

# Initialize model
embedder = AudioWatermarkEmbedder()

signal = tf.constant(np.random.randn(4, 1025, 45), dtype=tf.float32)
watermark = tf.constant(np.random.randn(4, 40), dtype=tf.float32)

# Forward pass
output = embedder((signal, watermark))
print(output.shape)  # (4, 1025, 45)


# Create the model (no need for self.build)
detector = WatermarkResNet()

# Generate a random batch of 8 spectrograms with shape (1025, 45, 1)
batch_size = 8
input_data = np.random.rand(batch_size, 1025, 45, 1).astype(np.float32)

# Run the model on this input
output = detector(input_data, training=False)

# Print the output
print("Output shape:", output.shape)       # Expected: (8, 40)
print("First output sample:", output[0].numpy())  # Values in [0, 1]