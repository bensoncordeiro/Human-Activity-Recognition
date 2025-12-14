import numpy as np
import pandas as pd

X = np.load("/content/X_windows.npy")
y = np.load("/content/y_labels.npy")
meta = pd.read_csv("/content/window_metadata.csv")

# boolean masks (splitting dataset based on person and then OS)
train_mask = meta["person"].isin(["Ben", "Dha"])
val_mask   = (meta["person"] == "Tan") & (meta["os"] == "ios")
test_mask  = (meta["person"] == "Tan") & (meta["os"] == "android")

X_train, y_train = X[train_mask.values], y[train_mask.values]
X_val,   y_val   = X[val_mask.values],   y[val_mask.values]
X_test,  y_test  = X[test_mask.values],  y[test_mask.values]

print(X_train.shape, X_val.shape, X_test.shape)

#Random Forest
#Step 1: Flattenning windows and extracting simple features
import numpy as np
import pandas as pd

def extract_features_batch(X_batch):
    """
    X_batch: [N, T, C]
    returns: [N, F] feature matrix (mean, std, min, max, energy per channel)
    """
    N, T, C = X_batch.shape
    feats = []
    for i in range(N):
        win = X_batch[i]
        row = []
        for c in range(C):
            x = win[:, c]
            row.extend([
                x.mean(),
                x.std(),
                x.min(),
                x.max(),
                np.sum(x**2),  # energy
            ])
        feats.append(row)
    return np.array(feats, dtype=float)

X_train_feat = extract_features_batch(X_train)
X_val_feat   = extract_features_batch(X_val)
X_test_feat  = extract_features_batch(X_test)

#Training and evluating Random Forest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# encoding string labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_val_int   = le.transform(y_val)
y_test_int  = le.transform(y_test)

# scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_val_scaled   = scaler.transform(X_val_feat)
X_test_scaled  = scaler.transform(X_test_feat)

# baseline RF
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_scaled, y_train_int)

print("Validation (Tan iOS) – Random Forest")
print(classification_report(y_val_int, rf.predict(X_val_scaled), target_names=le.classes_))
print("Test (Tan Android) – Random Forest")
print(classification_report(y_test_int, rf.predict(X_test_scaled), target_names=le.classes_))
print("Confusion matrix (test):")
print(confusion_matrix(y_test_int, rf.predict(X_test_scaled)))

#Model 2: 1D‑CNN on raw windows
import tensorflow as tf
from tensorflow.keras import layers, models

num_classes = len(le.classes_)

# reusing encoded integer labels
y_train_dl = y_train_int
y_val_dl   = y_val_int
y_test_dl  = y_test_int

input_shape = X_train.shape[1:]  # (256, 16)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv1D(64, kernel_size=5, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=5, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(256, kernel_size=3, activation="relu"),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train, y_train_dl,
    validation_data=(X_val, y_val_dl),
    epochs=30,
    batch_size=128,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )
    ]
)

print("CNN – Test (Tan Android)")
test_loss, test_acc = model.evaluate(X_test, y_test_dl)
print("Test accuracy:", test_acc)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class_names = le.classes_

# RF predictions
y_test_pred_rf = rf.predict(X_test_scaled)
cm_rf = confusion_matrix(y_test_int, y_test_pred_rf)

plt.figure(figsize=(5,4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest – Confusion Matrix (Test: Tan Android)")
plt.tight_layout()
plt.show()

print("Random Forest – Classification report (Test)")
print(classification_report(y_test_int, y_test_pred_rf, target_names=class_names))

# CNN predictions
y_test_pred_cnn = model.predict(X_test).argmax(axis=1)
cm_cnn = confusion_matrix(y_test_int, y_test_pred_cnn)

plt.figure(figsize=(5,4))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("1D‑CNN – Confusion Matrix (Test: Tan Android)")
plt.tight_layout()
plt.show()

print("1D‑CNN – Classification report (Test)")
print(classification_report(y_test_int, y_test_pred_cnn, target_names=class_names))

hist = history.history  # dict with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
epochs = range(1, len(hist["loss"]) + 1)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(epochs, hist["accuracy"], label="Train acc")
plt.plot(epochs, hist["val_accuracy"], label="Val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy per Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, hist["loss"], label="Train loss")
plt.plot(epochs, hist["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss per Epoch")
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score, f1_score

# overall metrics
rf_test_acc = accuracy_score(y_test_int, y_test_pred_rf)
rf_test_macro_f1 = f1_score(y_test_int, y_test_pred_rf, average="macro")

cnn_test_acc = accuracy_score(y_test_int, y_test_pred_cnn)
cnn_test_macro_f1 = f1_score(y_test_int, y_test_pred_cnn, average="macro")

results_df = pd.DataFrame({
    "Model": ["Random Forest", "1D‑CNN"],
    "Test accuracy (Tan Android)": [rf_test_acc, cnn_test_acc],
    "Test macro F1": [rf_test_macro_f1, cnn_test_macro_f1],
})
display(results_df)
print(results_df.to_markdown(index=False))
