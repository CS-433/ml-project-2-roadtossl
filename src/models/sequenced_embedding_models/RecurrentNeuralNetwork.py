import sys
import os
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

    def __init__(self, vocab_size=15000, word_embedding_dim=128, batch_size=8, epochs=10):
        """
        Initialize the RecurrentNeuralNetwork model

        Parameters:
        - vocab_size: int
            Size of the vocabulary
        - word_embedding_dim: int
            Dimension of the word embeddings
        - batch_size: int
            Number of samples per gradient update
        - epochs: int
            Number of epochs to train the model
        """

        # Save the hyperparameters as attributes
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        

    def predictions(self, model, dataset):
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

        # Print the hyperparameters
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Word embedding dimension: {self.word_embedding_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")

        # Load the training and testing data
        train_data = train_data.batch(self.batch_size)
        test_data = test_data.batch(self.batch_size)

        # Create the encoder
        print("Creating Encoder...(This may take a while)")
        encoder = tf.keras.layers.TextVectorization(max_tokens=self.vocab_size)
        encoder.adapt(train_data.map(lambda text, label: text))
        print(f"Encoder created with vocabulary size: {len(encoder.get_vocabulary())} ✔️")

        # Create the model
        print("Creating model...", end="\r")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            encoder,
            tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=self.word_embedding_dim, mask_zero=True), 
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.word_embedding_dim)),
            tf.keras.layers.Dense(self.word_embedding_dim, activation='relu'), 
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)])
        print("Model created ✔️   ")

        # Compile the model
        print("Compiling model...", end="\r")
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
            optimizer=tf.keras.optimizers.Adam(1e-4), 
            metrics=['accuracy'],
        )
        print("Model compiled ✔️   ")

        # Train the model
        print("Training model...", end = "\r")
        TRAIN_STEPS_PER_EPOCH = train_data_size // self.batch_size
        TEST_STEPS = test_data_size // self.batch_size

        model.fit(
            train_data.repeat(), 
            epochs=self.epochs,
            steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
            validation_data=test_data.repeat(), 
            validation_steps=TEST_STEPS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
        )
        print("Model trained ✔️   ")

        # Evaluate the model
        print("Evaluating model...")
        test_loss, test_acc = model.evaluate(test_data, steps=TEST_STEPS)

        # Print the test loss and accuracy
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_acc)

        # Path to the dataset
        DATASET_PATH = 'data/twitter-datasets/'
        TEST_SAMPLES_PATH = DATASET_PATH + 'test_data.txt'

        # Load the submission data
        submision_data = load_submission_data()

        # Predict the sentiment of the submission data
        predictions = self.predictions(model, submision_data)

        return predictions
