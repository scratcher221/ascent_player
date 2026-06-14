from __future__ import annotations

from typing import Sequence


def build_q_network(
    input_shape: Sequence[int],
    action_count: int,
    learning_rate: float,
):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=tuple(input_shape), name="frames")
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(action_count, name="q_values")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ascent_dqn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
    )
    return model
