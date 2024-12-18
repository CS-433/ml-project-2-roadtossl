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
        model = xgb.XGBClassifier(use_label_encoder=use_label_encoder, eval_metric=eval_metric)
        self.model = model

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        y_pred = [-1 if i == 0 else i for i in y_pred]
        return y_pred
