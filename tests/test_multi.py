import pytest
import tensorflow as tf

from multi_dense import MultiDense


class TestMultiDense:
    @pytest.fixture(scope="class")
    def activations(self):
        return ["relu", "tanh", "sigmoid"]

    @staticmethod
    def test_inheritance():
        assert issubclass(MultiDense, tf.keras.layers.Layer)

    def test_output_shape(self, activations):
        units = [4, 3, 2]
        layer = MultiDense(units, activations)

        x = tf.random.normal((5, 10))
        y = layer(x)

        expected_units = sum(units)
        assert y.shape == (5, expected_units)

    def test_activation_behavior(self, activations):
        units = [3, 10, 1]
        layer = MultiDense(units, activations)

        x = tf.random.normal((2, 5))
        y = layer(x)
        z = tf.matmul(x, layer.kernel) + layer.bias

        z_splits = tf.split(z, units, axis=-1)
        y_splits = tf.split(y, units, axis=-1)

        for z_split, y_split, activation in zip(z_splits, y_splits, activations):
            activation = tf.keras.activations.get(activation)
            tf.debugging.assert_near(y_split, activation(z_split))

    def test_gradients_exist(self, activations):
        layer = MultiDense([4, 3, 2], activations)

        x = tf.random.normal((3, 6))

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_sum(y)

        grads = tape.gradient(loss, layer.trainable_weights)
        for g, v in zip(grads, layer.trainable_weights):
            assert g is not None, f"No gradient for {v.name}"

    def test_integration_with_model(self, activations):
        inputs = tf.keras.Input(shape=(8,))
        x = MultiDense([4, 4, 4], activations)(inputs)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer="adam",
            loss="mse",
        )
        history = model.fit(
            tf.random.normal((16, 8)),
            tf.random.normal((16, 1)),
            epochs=1,
            verbose=0,
        )

        assert "loss" in history.history

    def test_serialization_roundtrip(self, activations):
        layer = MultiDense([3, 2, 1], activations)
        new_layer = MultiDense.from_config(layer.get_config())

        assert len(new_layer.units) == len(layer.units)
        assert all(
            new_units == units
            for new_units, units in zip(
                new_layer.units,
                layer.units,
            )
        )

        assert len(new_layer.activations) == len(layer.activations)
        assert all(
            new_activation == activation
            for new_activation, activation in zip(
                new_layer.activations,
                layer.activations,
            )
        )
