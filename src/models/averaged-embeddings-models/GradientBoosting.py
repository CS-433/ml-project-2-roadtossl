import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class GradientBoosting():
    """
    Gradient Boosting Classifier (using xgboost)
    """

    def __init__(self, use_label_encoder=False, eval_metric='logloss'):
        model = xgb.XGBClassifier(use_label_encoder=use_label_encoder, eval_metric=eval_metric)
        self.model = model

    def train(self, train_data, test_data, train_data_size, test_data_size):
        X_train, X_test, y_train, y_test = train_data[0], test_data[0], train_data[1], test_data[1]
    
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))