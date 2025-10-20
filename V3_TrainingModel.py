import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
# ------------------ Load embeddings and labels ------------------
embeddings = np.load(os.path.join("cache", "embeddings_final.npy"), allow_pickle=True)
print("embedding loaded", type(embeddings), embeddings.dtype)

# Convert object array of arrays into a 2D float32 array
embeddings = np.array(list(embeddings), dtype=np.float32)
print("converted embeddings:", embeddings.shape, embeddings.dtype)

# Optional: flatten if needed
embeddings = embeddings.reshape(embeddings.shape[0], -1)

labels = np.load(os.path.join("cache", "labels_final.npy"),allow_pickle=True)
print("labels loaded")
print(labels[0:5])
labels = labels.ravel()
print(labels[0:5])
labels= labels.astype(np.float32)
# ------------------ Train-test split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)

# ------------------ Define the model ------------------
input_size = X_train.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
model.summary()

# ------------------ Train the model ------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]


)

# ------------------ Plot training & validation loss ------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# ------------------ Evaluate model ------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# ------------------ Save the model ------------------
model.save("tensorflow_fake_news_model.h5")
print("Model saved as tensorflow_fake_news_model.h5")

# ------------------ Optional: predict ------------------
y_pred_prob = model.predict(X_test)
y_pred_class = (y_pred_prob > 0.5).astype(int)
