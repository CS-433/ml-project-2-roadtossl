import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import linear_model

class LogisticRegression():
    """
    Logistic Regression Classifier (using sklearn)
    """

    def __init__(self, max_iter=100000):
        """
        Initialize the LogisticRegression model

        Parameters:
        - max_iter: int
            Maximum number of iterations
        """

        # Initialize the model
        model = linear_model.LogisticRegression(max_iter=max_iter, verbose=True)

        # Save the model as an attribute
        self.model = model

    def train(self, X, y):
        """
        Train the LogisticRegression model and make predictions

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)

        # Print the classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Convert the predictions to the required format
        y_pred = [-1 if i == 0 else i for i in y_pred]

        return y_pred
