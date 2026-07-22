from __future__ import annotations

from typing import Sequence

StateInput = tuple  # (visual, vector) or visual only


def _advantage_center_layer():
    """Serializable mean-centering layer (avoids unsaved Lambda deserialization)."""
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="ascent")
    class AdvantageCenter(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            return super().get_config()

    return AdvantageCenter


def build_q_network(
    input_shape: Sequence[int],
    action_count: int,
    learning_rate: float,
    vector_dim: int = 0,
    *,
    dueling: bool = True,
    reason_count: int = 0,
):
    import tensorflow as tf

    AdvantageCenter = _advantage_center_layer()

    visual_input = tf.keras.Input(shape=tuple(input_shape), name="visual")
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu", name="conv1")(visual_input)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu", name="conv2")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu", name="conv3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)

    if vector_dim > 0:
        vector_input = tf.keras.Input(shape=(vector_dim,), name="vector")
        vector_branch = tf.keras.layers.Dense(128, activation="relu", name="vector_dense1")(
            vector_input
        )
        vector_branch = tf.keras.layers.Dense(128, activation="relu", name="vector_dense2")(
            vector_branch
        )
        x = tf.keras.layers.Concatenate(name="fuse")([x, vector_branch])
        inputs = [visual_input, vector_input]
    else:
        inputs = visual_input

    features = tf.keras.layers.Dense(512, activation="relu", name="features")(x)
    if dueling:
        value = tf.keras.layers.Dense(1, name="state_value")(features)
        advantage = tf.keras.layers.Dense(action_count, name="advantage")(features)
        advantage_centered = AdvantageCenter(name="advantage_centered")(advantage)
        q_values = tf.keras.layers.Add(name="q_values")([value, advantage_centered])
    else:
        q_values = tf.keras.layers.Dense(action_count, name="q_values")(features)

    outputs: list | object
    if reason_count > 0:
        reason_logits = tf.keras.layers.Dense(reason_count, name="reason_logits")(features)
        outputs = [q_values, reason_logits]
    else:
        outputs = q_values

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ascent_dqn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
    )
    return model
