import tensorflow as tf


class MultiDense(tf.keras.layers.Layer):
    def __init__(self, units, activations, **kwargs):
        super().__init__(**kwargs)

        if len(units) != len(activations):
            raise ValueError("units and activations must be the same length")

        self.units = list(units)
        self.activations = [
            tf.keras.activations.get(activation) for activation in activations
        ]

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        total_units = sum(self.units)
        self.w = self.add_weight(
            name="kernel",
            shape=(input_dim, total_units),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.b = self.add_weight(
            name="bias",
            shape=(total_units,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        z_splits = tf.split(z, self.units, axis=-1)
        a_splits = [activation(z) for activation, z in zip(self.activations, z_splits)]
        a = tf.concat(a_splits, axis=-1)
        return a

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activations": [
                    tf.keras.activations.serialize(activation)
                    for activation in self.activations
                ],
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        activations = config.pop("activations")
        config["activations"] = [
            tf.keras.activations.deserialize(activation) for activation in activations
        ]
        return cls(**config)
