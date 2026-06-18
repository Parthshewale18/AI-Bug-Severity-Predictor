import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load Data ------------------------------------------

df = pd.read_csv(r'../data/bug_reports_clean.csv')
print(df.shape)
print(df.isnull().sum())
df = df.dropna()

# Features and Target--------------------------------------

X = df['clean_text']
y = df['severity']

# Label Encoding ------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print("\nClasses:")
print(label_encoder.classes_)
# save label encoder
joblib.dump(label_encoder, "..\models\label_encoder.pkl")

# TF-IDF ---------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
X_tfidf = tfidf.fit_transform(X)
print("\nTF-IDF Shape:", X_tfidf.shape)


# Train Test Split ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(X_train.shape)
print(X_test.shape)

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    i: weights[i]  for i in range(len(weights))
}
print("Class Weights:")
print(class_weights)


# Convert SPARSE -> DENSE ------------------------------------

X_train = X_train.toarray().astype(np.float32)
X_test = X_test.toarray().astype(np.float32)


# Build DNN Model --------------------------------------------

print("Building Model")
model = Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(600, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(5, activation='softmax'),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# Callbacks -------------------------------------------------

earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('../models/bug_severity_dnn.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=64,
    class_weight=class_weights,
    callbacks = [earlystopping, checkpoint],
    verbose=1
)