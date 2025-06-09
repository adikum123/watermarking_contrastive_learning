import tensorflow as tf


class AudioWatermarkEmbedder(tf.keras.Model):
    """
    A U-Net style neural network for embedding watermarks into audio spectrograms.
    Input: (signal, watermark)
        - signal: 2D audio feature map, shape (B, 1025, 45)
        - watermark: 1D vector, shape (B, 40)
    Output:
        - watermarked signal, shape (B, 1025, 45)
    """

    def __init__(self, input_shape=(1025, 45), watermark_dim=40):
        super().__init__()
        self.input_shape_ = input_shape
        self.watermark_dim = watermark_dim

        # Project watermark into a 2D spatial map
        self.watermark_dense = tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation='relu')
        self.reshape_watermark = tf.keras.layers.Reshape(input_shape)

        # Encoder - Downsampling path
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))

        # Bottleneck - Encoded representation
        self.bottleneck = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')

        # Decoder - Upsampling path
        self.up2 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.up1 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')

        # Final projection to a single output channel
        self.final_conv = tf.keras.layers.Conv2D(1, (1, 1), activation='linear', padding='same')

    def call(self, inputs, training=False):
        # Unpack signal and watermark input tensors
        signal, watermark = inputs

        # Add channel dimension to signal and pad to even dimensions
        signal = tf.expand_dims(signal, axis=-1)  # (B, 1025, 45, 1)
        signal = tf.image.resize_with_crop_or_pad(signal, 1026, 46)

        # Expand watermark into spatial representation and match signal shape
        wm_embed = self.watermark_dense(watermark)  # (B, 1025*45)
        wm_embed = self.reshape_watermark(wm_embed)  # (B, 1025, 45)
        wm_embed = tf.expand_dims(wm_embed, axis=-1)  # (B, 1025, 45, 1)
        wm_embed = tf.image.resize_with_crop_or_pad(wm_embed, 1026, 46)

        # Concatenate signal and watermark along channel axis -> (B, 1026, 46, 2)
        x = tf.concat([signal, wm_embed], axis=-1)

        # Encoder path
        c1 = self.conv1(x)      # (B, 1026, 46, 32)
        p1 = self.pool1(c1)     # (B, 513, 23, 32)
        c2 = self.conv2(p1)     # (B, 513, 23, 64)
        p2 = self.pool2(c2)     # (B, 256, 11, 64)

        # Bottleneck
        b = self.bottleneck(p2)  # (B, 256, 11, 128)

        # Decoder path with skip connections
        u2 = self.up2(b)  # (B, 512, 22, 128)
        c2_crop = tf.image.resize_with_crop_or_pad(c2, tf.shape(u2)[1], tf.shape(u2)[2])
        c3 = self.conv3(tf.concat([u2, c2_crop], axis=-1))  # (B, 512, 22, 64)

        u1 = self.up1(c3)  # (B, 1024, 44, 64)
        c1_crop = tf.image.resize_with_crop_or_pad(c1, tf.shape(u1)[1], tf.shape(u1)[2])
        c4 = self.conv4(tf.concat([u1, c1_crop], axis=-1))  # (B, 1024, 44, 32)

        # Final projection -> (B, 1024, 44, 1)
        out = self.final_conv(c4)

        # Resize output to original input shape (B, 1025, 45, 1)
        out = tf.image.resize_with_crop_or_pad(out, 1025, 45)

        # Remove channel dim safely (result: B, 1025, 45)
        return tf.squeeze(out, axis=-1) if out.shape[-1] == 1 else out