from __future__ import annotations

from typing import Sequence

StateInput = tuple  # (visual, vector) or visual only

MODEL_VARIANT_NATURE = "nature"
MODEL_VARIANT_IMPALA_MID = "impala_mid"
MODEL_VARIANT_IMPALA_LARGE = "impala_large"
KNOWN_MODEL_VARIANTS = (
    MODEL_VARIANT_NATURE,
    MODEL_VARIANT_IMPALA_MID,
    MODEL_VARIANT_IMPALA_LARGE,
)


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


def _residual_block(x, channels: int, name: str):
    import tensorflow as tf

    shortcut = x
    y = tf.keras.layers.Conv2D(
        channels, 3, padding="same", activation="relu", name=f"{name}_conv1"
    )(x)
    y = tf.keras.layers.Conv2D(
        channels, 3, padding="same", activation=None, name=f"{name}_conv2"
    )(y)
    if shortcut.shape[-1] != channels:
        shortcut = tf.keras.layers.Conv2D(
            channels, 1, padding="same", activation=None, name=f"{name}_proj"
        )(shortcut)
    y = tf.keras.layers.Add(name=f"{name}_add")([shortcut, y])
    return tf.keras.layers.Activation("relu", name=f"{name}_relu")(y)


def _impala_stage(x, channels: int, name: str, *, blocks: int = 2):
    import tensorflow as tf

    x = tf.keras.layers.Conv2D(
        channels, 3, padding="same", activation="relu", name=f"{name}_stem"
    )(x)
    x = tf.keras.layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    for index in range(blocks):
        x = _residual_block(x, channels, f"{name}_b{index + 1}")
    return x


def _nature_visual_trunk(visual_input):
    import tensorflow as tf

    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu", name="conv1")(
        visual_input
    )
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu", name="conv2")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu", name="conv3")(x)
    return tf.keras.layers.Flatten(name="flatten")(x)


def _impala_visual_trunk(visual_input, *, large: bool = False):
    """Impala-style CNN: residual stages 32→64→128 (large: 64→128→256)."""
    import tensorflow as tf

    c1, c2, c3 = (64, 128, 256) if large else (32, 64, 128)
    blocks = 3 if large else 2
    x = _impala_stage(visual_input, c1, "impala_s1", blocks=blocks)
    x = _impala_stage(x, c2, "impala_s2", blocks=blocks)
    x = _impala_stage(x, c3, "impala_s3", blocks=blocks)
    return tf.keras.layers.Flatten(name="flatten")(x)


def _vector_branch(vector_input, *, width: int):
    import tensorflow as tf

    vector_branch = tf.keras.layers.Dense(width, activation="relu", name="vector_dense1")(
        vector_input
    )
    return tf.keras.layers.Dense(width, activation="relu", name="vector_dense2")(
        vector_branch
    )


def _feature_trunk(x, *, width: int, depth: int, use_layernorm: bool):
    import tensorflow as tf

    for index in range(max(1, depth)):
        x = tf.keras.layers.Dense(
            width, activation="relu", name=f"features{'' if index == 0 else index + 1}"
        )(x)
        if use_layernorm:
            x = tf.keras.layers.LayerNormalization(name=f"features_ln{index + 1}")(x)
    return x


def build_q_network(
    input_shape: Sequence[int],
    action_count: int,
    learning_rate: float,
    vector_dim: int = 0,
    *,
    dueling: bool = True,
    reason_count: int = 0,
    skill_count: int = 0,
    model_variant: str = MODEL_VARIANT_NATURE,
    use_mixed_precision: bool = False,
):
    import tensorflow as tf

    AdvantageCenter = _advantage_center_layer()
    variant = (model_variant or MODEL_VARIANT_NATURE).strip().lower()
    if variant not in KNOWN_MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model_variant={model_variant!r}; "
            f"expected one of {KNOWN_MODEL_VARIANTS}"
        )

    visual_input = tf.keras.Input(shape=tuple(input_shape), name="visual")
    if variant == MODEL_VARIANT_NATURE:
        x = _nature_visual_trunk(visual_input)
        vector_width = 128
        feature_width = 512
        feature_depth = 1
        use_layernorm = False
    elif variant == MODEL_VARIANT_IMPALA_MID:
        x = _impala_visual_trunk(visual_input, large=False)
        vector_width = 256
        feature_width = 1024
        feature_depth = 1
        use_layernorm = True
    else:
        x = _impala_visual_trunk(visual_input, large=True)
        vector_width = 256
        feature_width = 1024
        feature_depth = 2
        use_layernorm = True

    if vector_dim > 0:
        vector_input = tf.keras.Input(shape=(vector_dim,), name="vector")
        vector_branch = _vector_branch(vector_input, width=vector_width)
        x = tf.keras.layers.Concatenate(name="fuse")([x, vector_branch])
        inputs = [visual_input, vector_input]
    else:
        inputs = visual_input

    features = _feature_trunk(
        x, width=feature_width, depth=feature_depth, use_layernorm=use_layernorm
    )

    # Keep Q / aux heads in float32 for mixed-precision stability.
    head_dtype = "float32" if use_mixed_precision else None
    if dueling:
        value = tf.keras.layers.Dense(1, name="state_value", dtype=head_dtype)(features)
        advantage = tf.keras.layers.Dense(
            action_count, name="advantage", dtype=head_dtype
        )(features)
        advantage_centered = AdvantageCenter(name="advantage_centered")(advantage)
        q_values = tf.keras.layers.Add(name="q_values")([value, advantage_centered])
    else:
        q_values = tf.keras.layers.Dense(
            action_count, name="q_values", dtype=head_dtype
        )(features)

    outputs: list = [q_values]
    if reason_count > 0:
        reason_logits = tf.keras.layers.Dense(
            reason_count, name="reason_logits", dtype=head_dtype
        )(features)
        outputs.append(reason_logits)
    if skill_count > 0:
        skill_logits = tf.keras.layers.Dense(
            skill_count, name="skill_logits", dtype=head_dtype
        )(features)
        outputs.append(skill_logits)
    if len(outputs) == 1:
        outputs = outputs[0]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"ascent_dqn_{variant}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
    )
    return model
