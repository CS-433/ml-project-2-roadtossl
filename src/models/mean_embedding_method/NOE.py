import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

tf.disable_progress_bar()

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

# Parameters
BUFFER_SIZE = 1000000
BATCH_SIZE = 16
VOCAB_SIZE = 1000
EPOCHS = 10

# Path to data
data_dir = "data/twitter-datasets"  # Adjust based on your file structure

# Determine split sizes without consuming the datasets
neg_size = sum(1 for _ in tf.data.TextLineDataset(os.path.join(data_dir, "train_neg.txt")))
pos_size = sum(1 for _ in tf.data.TextLineDataset(os.path.join(data_dir, "train_pos.txt")))

print(f"Negative samples: {neg_size}, Positive samples: {pos_size}")

# Create datasets for negative and positive samples
neg_dataset = tf.data.TextLineDataset(os.path.join(data_dir, "train_neg.txt")).map(lambda x: (x, 0))
pos_dataset = tf.data.TextLineDataset(os.path.join(data_dir, "train_pos.txt")).map(lambda x: (x, 1))

# Shuffle datasets
neg_dataset = neg_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
pos_dataset = pos_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

# Determine split sizes
neg_train_size = int(0.8 * neg_size)
pos_train_size = int(0.8 * pos_size)

# Split datasets
neg_train = neg_dataset.take(neg_train_size)
neg_test = neg_dataset.skip(neg_train_size)
pos_train = pos_dataset.take(pos_train_size)
pos_test = pos_dataset.skip(pos_train_size)

# Combine datasets
train_dataset = neg_train.concatenate(pos_train)
test_dataset = neg_test.concatenate(pos_test)

# Count samples
num_train_samples = neg_train_size + pos_train_size
num_test_samples = neg_size + pos_size - num_train_samples

print(f"Training samples: {num_train_samples}, Testing samples: {num_test_samples}")

# Shuffle and batch datasets
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Prepare the TextVectorization layer
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

# Verify
print("Vocabulary Size:", len(encoder.get_vocabulary()))
print("Sample Vocabulary:", encoder.get_vocabulary()[:10])

# Build the model
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)

# Calculate steps per epoch
train_steps_per_epoch = num_train_samples // BATCH_SIZE
validation_steps = num_test_samples // BATCH_SIZE

# Train the model
history = model.fit(
    train_dataset.repeat(),
    epochs=EPOCHS,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=test_dataset.repeat(),
    validation_steps=validation_steps
)

# Evaluate the model
test_loss, test_acc = model.evaluate(
    test_dataset,
    steps=validation_steps
)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Plot the results
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()