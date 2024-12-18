import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.utils.dataloader import load_data_mean
from sklearn.linear_model import LogisticRegression

class LogisticRegressionM():
    """
    Logistic Regression Classifier (using sklearn)
    """

    def __init__(self, max_iter=100000):
        model = LogisticRegression(max_iter=max_iter)
        self.model = model

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

X, y = load_data_mean(full=True)
model = LogisticRegressionM()
model.train(X, y)