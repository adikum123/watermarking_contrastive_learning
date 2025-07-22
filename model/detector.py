import tensorflow as tf
from tensorflow.keras import layers


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters, stride=1):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.shortcut_conv = None
        self.shortcut_bn = None
        if stride != 1:
            self.shortcut_conv = layers.Conv2D(filters, 1, strides=stride, padding='same')
            self.shortcut_bn = layers.BatchNormalization()

        self.relu = layers.ReLU()

    def call(self, x, training=False):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.shortcut_conv:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)

        x += shortcut
        return self.relu(x)

class WatermarkResNet(tf.keras.Model):

    def __init__(self, input_shape=(1025, 45, 1), num_blocks=4, output_bits=40):
        super().__init__()
        self.initial_conv = layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')
        self.initial_bn = layers.BatchNormalization()

        self.res_blocks = [ResidualBlock(32) for _ in range(num_blocks)]

        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(output_bits, activation='sigmoid')  # binary output

        # Build the model by calling with a dummy input (for summary & saving)
        self.build((None,) + input_shape)

    def call(self, inputs, training=False):
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        for block in self.res_blocks:
            x = block(x, training=training)

        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)