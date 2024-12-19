import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.utils.dataloader import load_data_mean

class GradientBoosting():
    """
    Gradient Boosting Classifier (using xgboost)
    """

    
    def __init__(self, use_label_encoder=False, eval_metric='logloss'):
        """
        Initialize the GradientBoosting model

        Parameters:
        - use_label_encoder: bool
            Whether to use label encoder
        - eval_metric: str
            Evaluation metric to use
        """

        # Initialize the model
        model = xgb.XGBClassifier(use_label_encoder=use_label_encoder, eval_metric=eval_metric)

        # Save the model as an attribute
        self.model = model

    def train(self, X, y):
        """
        Train the GradientBoosting model and make predictions

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

