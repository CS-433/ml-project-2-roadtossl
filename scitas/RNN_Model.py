import os
os.system('nvidia-smi')
import numpy as np
import tensorflow as tf
import pandas as pd
from utils.dataloader import load_data_seq, load_submission_data
from utils.submission import write_submission

# Print TensorFlow version and device information
print("TensorFlow version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("List of devices:", tf.config.list_physical_devices())

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU found. Running on CPU")

def predict(model, dataset):
    """
    Predict the binary sentiment of text data from a dataset.
    """
    texts = list(dataset.as_numpy_iterator())
    input_batch = tf.constant(texts)
    probabilities = model.predict(input_batch)
    binary_predictions = tf.where(probabilities < 0.5, -1, 1).numpy()
    return binary_predictions

print("Loading data...")
train_data, test_data, train_data_size, test_data_size = load_data_seq(full=True)
print("Data loaded.")

VOCAB_SIZE = 5000
BATCH_SIZE = 32
EPOCHS = 10
WORD_EMBEDDING_DIM = 128

# Enable prefetching for better GPU utilization
train_data = train_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Creating encoder...")
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_data.map(lambda text, label: text))
print("Encoder created.")

# Model creation
print("Creating model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.string),
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()), 
        output_dim=WORD_EMBEDDING_DIM, 
        mask_zero=True
    ),
    tf.keras.layers.Dropout(0.5, seed=42),  # Added fixed seed
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        WORD_EMBEDDING_DIM,
        return_sequences=False
    )),
    tf.keras.layers.Dense(WORD_EMBEDDING_DIM, activation='relu'),
    tf.keras.layers.Dropout(0.5, seed=42),  # Added fixed seed
    tf.keras.layers.Dense(1)
])

# Model compilation
print("Compiling model...")
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy'],
)
print("Model compiled.")

TRAIN_STEPS_PER_EPOCH = train_data_size // BATCH_SIZE
TEST_STEPS = test_data_size // BATCH_SIZE

# Add callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=1,
        min_lr=1e-6
    )
]

# Training
history = model.fit(
    train_data.repeat(),
    epochs=EPOCHS,
    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
    validation_data=test_data.repeat(),
    validation_steps=TEST_STEPS,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(test_data, steps=TEST_STEPS)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Load and predict on submission data
submission_data = load_submission_data()
predictions = predict(model, submission_data)
write_submission(predictions, 'submission2.csv')