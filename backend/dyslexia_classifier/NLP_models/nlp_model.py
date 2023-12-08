# NLP model implementation for text classification
# Example: Dummy model for illustration purposes
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from utils import plot_confusion_matrix

# for other files, use -> from nlp_model import NLPModel

class NLPModel:
    def __init__(self, input_shape, labels_dict):
        self.input_shape = input_shape
        self.labels_dict = labels_dict
        self.model = self.build_model()
        self.es = EarlyStopping(patience=10, restore_best_weights=True)

    def build_model(self):
        model = Sequential([
            layers.LSTM(105, input_shape=self.input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(6, activation='softmax'),
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])
        return model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train,
                       validation_split=0.3,
                       epochs=50,
                       batch_size=32,
                       callbacks=[self.es])

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test loss: {loss:.4f}')
        print(f'Test accuracy: {accuracy:.4f}')

    def predict(self, X):
        predictions = self.model.predict(X)
        class_labels_predictions = np.argmax(predictions, axis=1)
        return class_labels_predictions

    def plot_confusion_matrix(self, y_true, y_pred):
        plot_confusion_matrix(y_true, y_pred, self.labels_dict)
