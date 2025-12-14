import numpy as np
import matplotlib.pyplot as plt

# Load your data
X_test = np.load("DATA/X_windows.npy")
y_test = np.load("DATA/y_labels.npy")
model = ...  # your trained CNN model

# Pick one example window from the test set
# Let's pick a "running" example that the model predicts correctly
idx = 100  # or any index you want to show

example_window = X_test[idx]  # shape: (256, 16)
true_label = y_test[idx]
pred_probs = model.predict(example_window.reshape(1, 256, 16))[0]
pred_label = le.classes_[np.argmax(pred_probs)]
pred_confidence = np.max(pred_probs)

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# LEFT: Input – accelerometer x, y, z
ax1 = axes[0]
time = np.arange(256)
ax1.plot(time, example_window[:, 0], label="Acc X", linewidth=1.5)
ax1.plot(time, example_window[:, 1], label="Acc Y", linewidth=1.5)
ax1.plot(time, example_window[:, 2], label="Acc Z", linewidth=1.5)
ax1.set_xlabel("Time (samples @ 100 Hz)")
ax1.set_ylabel("Acceleration (m/s²)")
ax1.set_title(f"INPUT: 2.56-second sensor window (Accelerometer X, Y, Z)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# RIGHT: Output – model prediction
ax2 = axes[1]
class_names = le.classes_
colors = ["green" if c == pred_label else "lightgray" for c in class_names]
bars = ax2.bar(class_names, pred_probs, color=colors, edgecolor="black", linewidth=2)
ax2.set_ylabel("Model Confidence")
ax2.set_ylim([0, 1])
ax2.set_title(f"OUTPUT: CNN Prediction → {pred_label.upper()} ({pred_confidence:.1%} confidence)")
ax2.grid(True, axis="y", alpha=0.3)

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("example_input_output.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"True label: {true_label}")
print(f"Predicted label: {pred_label} ({pred_confidence:.1%})")
