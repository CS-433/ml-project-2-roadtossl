import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import tensorflow as tf

from src.utils.dataloader import load_data_seq, load_submission_data
from src.utils.submission import write_submission
from sklearn.model_selection import train_test_split
from src.utils.dataloader import load_data_seq

class RecurrentNeuralNetwork():
    """
    Recurrent Neural Network Classifier (using TensorFlow)
    """

    def __init__(self, vocab_size=15000, word_embedding_dim=128, batch_size=64, epochs=10):
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        

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

    def train(self, train_data, test_data, train_data_size, test_data_size):

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Word embedding dimension: {self.word_embedding_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")

        print("Creating Encoder...")
        encoder = tf.keras.layers.TextVectorization(max_tokens=self.vocab_size)
        encoder.adapt(train_data.map(lambda text, label: text))
        print("Encoder created.")
        print(f"    Vocabulary size: {len(encoder.get_vocabulary())}")

        print("Creating model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            encoder,
            tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=self.word_embedding_dim, mask_zero=True), 
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.word_embedding_dim)),
            tf.keras.layers.Dense(self.word_embedding_dim, activation='relu'), 
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)])
        print("Model created.")

        print("Compiling model...")
        # Model compilation
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
            optimizer=tf.keras.optimizers.Adam(1e-4), 
            metrics=['accuracy'],
        )
        print("Model compiled.")

        TRAIN_STEPS_PER_EPOCH = train_data_size // self.batch_size
        TEST_STEPS = test_data_size // self.batch_size

        print("Training model...")
        model.fit(
            train_data.repeat(), 
            epochs=self.epochs,
            steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
            validation_data=test_data.repeat(), 
            validation_steps=TEST_STEPS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
        )
        print("Model trained.")

        print("Evaluating model...")
        test_loss, test_acc = model.evaluate(test_data, steps=TEST_STEPS)

        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)

        DATASET_PATH = 'data/twitter-datasets/'
        TEST_SAMPLES_PATH = DATASET_PATH + 'test_data.txt'

        # Load the submission data
        submision_data = load_submission_data()

        # Predict the sentiment of the submission data
        predictions = predict(model, submision_data)

        write_submission(predictions, '../../../data/submission/submission.csv')

model: RecurrentNeuralNetwork = RecurrentNeuralNetwork(vocab_size=10000)
train_data, test_data, train_data_size, test_data_size = load_data_seq(False)
model.train(train_data, test_data, train_data_size, test_data_size)