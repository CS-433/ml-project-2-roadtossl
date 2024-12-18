import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SupportVectorMachine():
    """
    Support Vector Machine Classifier (using sklearn)
    """

    def __init__(self, kernel='rbf', random_state=42):
        model = SVC(kernel=kernel, random_state=random_state, verbose=True)
        self.model = model

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))