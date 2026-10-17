"""Keras model selection loop."""
import tensorflow as tf

EPOCHS = 30
x_train = tf.random.uniform((100, 8))
y_train = tf.random.uniform((100, 1))

for units in [16, 32, 64]:  # expect: ML.TENSORFLOW.MEMORY_RELEASE_CHECKER
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2)  # expect: ML.TENSORFLOW.CHECKPOINTING_CHECKER, ML.TENSORFLOW.EARLY_STOPPING_CHECKER

restored = tf.keras.models.load_model("best_model.keras")  # expect: ML.TENSORFLOW.EXCEPTION_HANDLING_CHECKER


@tf.function  # expect: ML.MISSING_DOCSTRING
def collect(xs):
    out = []
    for x in xs:
        out.append(x * 2)  # expect: ML.TENSORFLOW.TENSOR_ARRAY_CHECKER
    return out

# File-level findings (reported at line 0):
# expect-file: ML.TENSORFLOW.BATCH_NORMALISATION_CHECKER
# expect-file: ML.TENSORFLOW.DEPENDENT_THRESHOLD_CHECKER
# expect-file: ML.TENSORFLOW.DROPOUT_USAGE_CHECKER
# expect-file: ML.TENSORFLOW.LEARNING_RATE_SCHEDULER_CHECKER
# expect-file: ML.TENSORFLOW.LOGGING_CHECKER
# expect-file: ML.TENSORFLOW.MODEL_EVALUATION_CHECKER
# expect-file: ML.TENSORFLOW.RANDOMNESS_CONTROL_CHECKER
