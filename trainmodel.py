from SignLanguageDetectionUsingML.features import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

# label_map = {label:num for num, label in enumerate(actions)}
# sequences, labels = [], []
# for action in actions:
#     for sequence in range(1, no_sequences+1):
#         window = []
#         for frame_num in range(1, sequence_length+1):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])

print("Loading dataset instantly...")
X = np.load("X_dataset.npy")
y = np.load("y_dataset.npy")
print("Loaded!")

# X = np.array(sequences)
# y = to_categorical(labels).astype(int)
y = to_categorical(y).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
noise = np.random.normal(0, 0.03, X_train.shape)  # 0.01 ~ 1% jitter
X_train = X_train + noise

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#For  LSTM model(video sequences))
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
# res = [.7, 0.2, 0.1]

model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=300, callbacks=[tb_callback, early_stop])
model.summary()
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')