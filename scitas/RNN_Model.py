import numpy as np
import tensorflow as tf
import pandas as pd

# Add GPU configuration code at the start
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     try:
#         # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
#         for device in physical_devices:
#             tf.config.experimental.set_memory_growth(device, True)
#         print(f"Found {len(physical_devices)} GPU(s). GPU acceleration enabled.")
#     except RuntimeError as e:
#         print(f"Error configuring GPU devices: {e}")
# else:
#     print("No GPU devices found. Running on CPU.")


from utils.dataloader import load_data, load_submission_data
from utils.submission import write_submission

def predict(model, dataset):
    """
    Predict the binary sentiment of text data from a dataset.

    Args:
        model (tf.keras.Model): Trained model with TextVectorization layer.
        dataset (tf.data.Dataset): Dataset containing text samples.

    Returns:
        numpy.ndarray: Predicted binary sentiments (-1 or 1) for each input.
    """
    # Extract text from dataset
    texts = list(dataset.as_numpy_iterator())
    input_batch = tf.constant(texts)

    # Perform predictions
    probabilities = model.predict(input_batch)

    # Convert probabilities to binary predictions (-1 or 1)
    binary_predictions = [-1 if prob < 0.5 else 1 for prob in probabilities]
    #binary_predictions = np.where(probabilities < 0, -1, 1)

    return binary_predictions

train_data, test_data, train_data_size, test_data_size = load_data(full=True)

VOCAB_SIZE = 15000
BATCH_SIZE = 8
EPOCHS = 10
WORD_EMBEDDING_DIM = 128

encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_data.map(lambda text, label: text))

# Model creation
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.string),  # Add explicit input layer
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=WORD_EMBEDDING_DIM, mask_zero=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(WORD_EMBEDDING_DIM)),
    tf.keras.layers.Dense(WORD_EMBEDDING_DIM, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)])

# Model compilation
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy'],
)

TRAIN_STEPS_PER_EPOCH = train_data_size // BATCH_SIZE
TEST_STEPS = test_data_size // BATCH_SIZE

history = model.fit(
    train_data.repeat(),
    epochs=EPOCHS,
    steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
    validation_data=test_data.repeat(),
    validation_steps=TEST_STEPS,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)

test_loss, test_acc = model.evaluate(test_data, steps=TEST_STEPS)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

DATASET_PATH = 'data/twitter-datasets/'
TEST_SAMPLES_PATH = DATASET_PATH + 'test_data.txt'

# Load the submission data
submision_data = load_submission_data()

# Predict the sentiment of the submission data
predictions = predict(model, submision_data)

write_submission(predictions, 'data/submission/submission.csv')
