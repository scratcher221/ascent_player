from __future__ import annotations

from typing import Sequence

StateInput = tuple  # (visual, vector) or visual only


def build_q_network(
    input_shape: Sequence[int],
    action_count: int,
    learning_rate: float,
    vector_dim: int = 0,
):
    import tensorflow as tf

    visual_input = tf.keras.Input(shape=tuple(input_shape), name="visual")
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(visual_input)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)

    if vector_dim > 0:
        vector_input = tf.keras.Input(shape=(vector_dim,), name="vector")
        vector_branch = tf.keras.layers.Dense(64, activation="relu")(vector_input)
        x = tf.keras.layers.Concatenate()([x, vector_branch])
        inputs = [visual_input, vector_input]
    else:
        inputs = visual_input

    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(action_count, name="q_values")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ascent_dqn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
    )
    return model
