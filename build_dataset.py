from SignLanguageDetectionUsingML.features import *
import numpy as np
import os

# Label mapping
label_map = {label:num for num, label in enumerate(actions)}

X = []
y = []

print("Building dataset... This will take time ONLY ONCE.")

for action in actions:
    print(f"Processing action: {action}")
    for sequence in range(1, no_sequences + 1):

        npy_path = os.path.join(DATA_PATH, action, str(sequence), f"1.npy")

        if not os.path.exists(npy_path):
            print("Missing:", npy_path)
            continue

        keypoints = np.load(npy_path)
        X.append(keypoints)
        y.append(label_map[action])

X = np.array(X)
y = np.array(y)

print("Saving dataset...")
np.save("X_dataset.npy", X)
np.save("y_dataset.npy", y)

print("Dataset saved successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
