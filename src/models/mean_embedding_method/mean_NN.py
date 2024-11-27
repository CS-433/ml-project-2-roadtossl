from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn_model import SKLearnModel

class ClassifierAverageNN(SKLearnModel):
    """
    Neural Network Classifier (using sklearn)
    """

    def __init__(self, hidden_layer_sizes=(100,), max_iter=200, random_state=42):
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state, verbose=True)
        super().__init__(model)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print("Training neural network...")
        with tqdm(total=self.model.max_iter) as pbar:

            def update_progress(iter_num, loss):
                pbar.update(1)
                pbar.set_description(f'Loss: {loss:.4f}')

            self.model._callback = update_progress
            self.model.fit(X_train, y_train)
        
        
        # Post-training Evaluation
        y_pred = self.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
