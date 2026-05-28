from queue import Queue
import subprocess

from features import *
import time
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from collections import Counter, deque
import pyttsx3

last_hand_time = time.time()
timeout = 5

# Detection variables
sentence = []
accuracy = []
predictions = deque(maxlen=15)

threshold = 0.75
last_spoken_gesture = None
last_spoken_time = 0
stable_candidate = None
last_accepted_gesture = False
stable_count = 0
STABLE_FRAMES_REQUIRED = 8
speak_delay = 0.5  # tighter response
mismatch_count = 0
MAX_MISMATCH = 2

# Word building variables
current_word = ""
all_words = []
space_added = False          # prevent multiple spaces
last_gesture_time = time.time()

last_speech_time = 0
SPEECH_COOLDOWN = 1.0

# Load model
with open("model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("model.h5")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
speech_queue = Queue()

colors = [(245,117,16) for _ in range(20)]

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def speak_gesture(text):
    global last_speech_time

    current_time = time.time()

    if current_time - last_speech_time < SPEECH_COOLDOWN:
        return

    last_speech_time = current_time

    safe_text = text.replace("'", "")

    command = [
        "powershell",
        "-Command",
        f"""
        Add-Type -AssemblyName System.Speech;
        $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
        $speak.Speak('{safe_text}');
        """
    ]

    subprocess.Popen(
        command,
        creationflags=subprocess.CREATE_NO_WINDOW
    )



cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.65, # Adjusted for better detection
    min_tracking_confidence=0.7) as hands: #Adjusted for better tracking/movement

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        #Active region border
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0,40), (300,400), 255, 2)
        cv2.putText(frame, "Active Region", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        image, results = mediapipe_detection(cropframe, hands)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(cropframe, handLms, mp_hands.HAND_CONNECTIONS)

        current_time = time.time()
        keypoints = extract_keypoints(results)

        if keypoints is not None and np.any(keypoints):
            space_added = False # reset space flag on new hand
            last_hand_time = current_time
            last_gesture_time = current_time

            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(res)
            confidence = res[predicted_idx]

            predictions.append(predicted_idx)  # keep for display smoothing if needed
            most_common_idx = Counter(predictions).most_common(1)[0][0]
            # USE DIRECT PREDICTION (NO most_common for speech)
            if confidence > threshold:
                current_gesture = actions[most_common_idx]
                if stable_candidate is None:
                    stable_candidate = current_gesture
                #Track stable frames
                if current_gesture == stable_candidate:
                    stable_count +=1
                    mismatch_count = 0
                else:
                    mismatch_count += 1
                    if mismatch_count >= MAX_MISMATCH:
                        stable_candidate = current_gesture
                        stable_count = 1
                        mismatch_count = 0

                print(f"Gesture: {current_gesture} | stable_count: {stable_count} | last_spoken: {last_spoken_gesture}")

                # Update display
                if len(sentence) == 0 or current_gesture != sentence[-1]:
                    sentence.append(current_gesture)
                    accuracy.append(f"{confidence*100:.2f}")

                # REAL FIXED SPEAK LOGIC
                if (stable_count >= STABLE_FRAMES_REQUIRED and current_gesture != last_accepted_gesture):
                    print(f"SPEAKING: {current_gesture}")
                    speak_gesture(current_gesture)
                    current_word += current_gesture
                    space_added = False
                    last_spoken_gesture = current_gesture
                    last_accepted_gesture = current_gesture
                    stable_count = 0

            if len(sentence) > 1:
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]

        else:
            if (current_time - last_hand_time) > 1.5:
                if current_word and not space_added:
                    all_words.append(current_word)
                    speak_gesture("space")
                    print(f"Word completed: {current_word}")
                    current_word = ""
                    space_added = True
                last_spoken_gesture = None
                stable_candidate = None
                stable_count = 0
                last_accepted_gesture = None
            if current_time - last_hand_time > timeout:
                predictions.clear()

        # Display output
        cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: " + ' '.join(sentence) + "  " + ''.join(accuracy),
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        display_text = ' '.join(all_words)
        if current_word:
            display_text += (" " if all_words else "") + current_word
        
        words_on_screen = str(display_text[-20:] if len(display_text) > 20 else display_text)

        if not words_on_screen:
            words_on_screen = " "
        # h, w = int(frame.shape[0]), int(frame.shape[1])
        cv2.rectangle(frame, (0, 430), (640, 480), (30, 30, 30), -1)
        cv2.putText(frame, words_on_screen,
                    (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        # print(type(words_on_screen), repr(words_on_screen))
        frame[40:400, 0:300] = cropframe
        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()