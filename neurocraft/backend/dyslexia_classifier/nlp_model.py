# NLP model implementation for text classification
# Example: Dummy model for illustration purposes


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import l1_l2, l2

# gpu is slower for this challenge
# Disable GPUs:
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# for other files, use -> from nlp_model import NLPModel

class NLPModel:
    def __init__(self, X, y, input_shape):
        self.X = X
        self.y = y
        self.input_shape = input_shape
        self.model = self.build_model(input_shape)
        self.es = EarlyStopping(patience=10, restore_best_weights=True)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        self.lr_scheduler = LearningRateScheduler(self.scheduler)

    def scheduler(self, epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    def build_model(self, input_shape):
        input_nlp = layers.Input(shape=input_shape)
        x = layers.Bidirectional(layers.LSTM(32, kernel_constraint=MaxNorm(3), return_sequences=True, kernel_regularizer=l2(0.01)))(input_nlp)
        x = layers.Dropout(0.5)(x)
        x = layers.Bidirectional(layers.LSTM(16, kernel_constraint=MaxNorm(3), kernel_regularizer=l2(0.01)))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(16, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Dropout(0.5)(x)
        output_nlp = layers.Dense(16, activation="relu")(x)
        output_nlp = layers.Dense(3, activation="softmax")(output_nlp)
        model = models.Model(inputs=input_nlp, outputs=output_nlp)

        model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def train(self):
        history = self.model.fit(
            self.X, self.y,
            validation_split=0.3,
            epochs=50,
            batch_size=32,
            callbacks=[self.es, self.reduce_lr, self.lr_scheduler]
        )
        print(f"✅ Model trained on {len(self.X)} rows with max val accuracy: {round(np.max(history.history['val_accuracy']), 2)}")

        return self.model, history

    def evaluate(self, X_test_text, y_test):
        loss, accuracy = self.model.evaluate(X_test_text, y_test)
        print(f'Test loss: {loss:.4f}')
        print(f'Test accuracy: {accuracy:.4f}')

    def predict(self, text):
        predictions = self.model.predict(text)
        class_labels_predictions = np.argmax(predictions, axis=1)
        return class_labels_predictions
