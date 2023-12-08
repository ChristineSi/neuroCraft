# Combine model implementation for text classification
# Example: Dummy model for illustration purposes
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from neurocraft.backend.dyslexia_classifier.NLP_models.nlp_model import NLPModel
from neurocraft.backend.dyslexia_classifier.NLP_models.num_model import NumModel
from neurocraft.utils import plot_confusion_matrix

class CombinedModel:
    def __init__(self, X_train_pad, input_dim, labels_dict):
        self.nlp_model = NLPModel(X_train_pad, labels_dict)
        self.num_model = NumModel(input_dim, labels_dict)
        self.labels_dict = labels_dict
        self.es = EarlyStopping(patience=10, restore_best_weights=True)
        self.model = self.build_model()

    def build_model(self):
        input_text = self.nlp_model.nlp_model.input
        output_text = self.nlp_model.nlp_model.output
        input_num = self.num_model.num_model.input
        output_num = self.num_model.num_model.output

        combined = layers.concatenate([output_text, output_num])
        x = layers.Dense(32, activation="relu")(combined)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(6, activation="softmax")(x)

        model = models.Model(inputs=[input_text, input_num], outputs=outputs)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])
        return model

    def train(self, X_train_pad, X_train_num, y_train):
        self.model.fit(x=[X_train_pad, X_train_num], y=y_train,
                       validation_split=0.3,
                       epochs=100,
                       batch_size=32,
                       callbacks=[self.es])

    def evaluate(self, X_test_pad, X_test_num, y_test):
        loss, accuracy = self.model.evaluate([X_test_pad, X_test_num], y_test)
        print(f'Test loss: {loss:.4f}')
        print(f'Test accuracy: {accuracy:.4f}')

    def predict(self, X_test_pad, X_test_num):
        predictions = self.model.predict([X_test_pad, X_test_num])
        class_labels_predictions = np.argmax(predictions, axis=1)
        return class_labels_predictions

    def plot_confusion_matrix(self, y_true, y_pred):
        plot_confusion_matrix(y_true, y_pred, self.labels_dict)
