import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.utils.dataloader import load_data_mean
from tqdm import tqdm

class NeuralNetwork():
    """
    Neural Network Classifier (using sklearn)
    """

    def __init__(self, hidden_layer_sizes=(100,), max_iter=200, random_state=42):
        """
        Initialize the NeuralNetwork model

        Parameters:
        - hidden_layer_sizes: tuple
            The ith element represents the number of neurons in the ith hidden layer
        - max_iter: int
            Maximum number of iterations
        - random_state: int
            Random seed for reproducibility
        """

        # Initialize the model
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state, verbose=True)

        # Save the model as an attribute
        self.model = model

    def train(self, X, y):
        """
        Train the NeuralNetwork model and make predictions
        
        Parameters:
        - X: np.ndarray
            Features
        - y: np.ndarray
            Labels
        
        Returns:
        - y_pred: np.ndarray
            Predictions
        """

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train the model
        print("Training neural network...")
        with tqdm(total=self.model.max_iter) as pbar:

            def update_progress(iter_num, loss):
                """
                Update the progress bar with the current loss
                
                Parameters:
                - iter_num: int
                    The current iteration number

                - loss: float
                    The current loss
                """
                pbar.update(1)
                pbar.set_description(f'Loss: {loss:.4f}')

            # Train the model
            self.model._callback = update_progress
            self.model.fit(X_train, y_train)
        
        
        # Make predictions
        y_pred = self.model.predict(X_test)

        # Print the classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Convert the predictions to the required format
        y_pred = [-1 if i == 0 else i for i in y_pred]

        return y_pred
