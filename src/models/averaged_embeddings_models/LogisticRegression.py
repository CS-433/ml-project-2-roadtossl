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
        model = linear_model.LogisticRegression(max_iter=max_iter, verbose=True)
        self.model = model

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        y_pred = [-1 if i == 0 else i for i in y_pred]
        return y_pred
