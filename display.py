from features import *
import time
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from collections import Counter, deque

last_hand_time = time.time()
timeout = 5
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

colors = []
for i in range(0,20):
    colors.append((245,117,16))
print(len(colors))

def prob_viz(res, actions, input_frame, colors,threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# 1. New detection variables
sequence = []
sentence = []
accuracy=[]
predictions = deque(maxlen=15)
threshold = 0.6 

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        cropframe=frame[40:400,0:300]
        # print(frame.shape)
        frame=cv2.rectangle(frame,(0,40),(300,400),255,2)
        # frame=cv2.putText(frame,"Active Region",(75,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,255,2)
        cv2.putText(frame,"Active Region",(10,35), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        image, results = mediapipe_detection(cropframe, hands)
        # print(results)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(cropframe, handLms, mp_hands.HAND_CONNECTIONS)

        # 2. Prediction logic
        currtent_time = time.time()
        keypoints = extract_keypoints(results)
        if keypoints is not None and np.any(keypoints):
            last_hand_time = currtent_time
            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            #3. Viz logic
            most_common = Counter(predictions).most_common(1)[0][0]
            if most_common == np.argmax(res) and res[most_common] > threshold:
                if len(sentence) == 0 or actions[most_common] != sentence[-1]:
                    sentence.append(actions[most_common])
                    accuracy.append(f"{res[most_common]*100:.2f}")
                # if np.unique(predictions[-10:])[0]==np.argmax(res): 
                #     if res[np.argmax(res)] > threshold: 
                #         if len(sentence) > 0: 
                #             if actions[np.argmax(res)] != sentence[-1]:
                #                 sentence.append(actions[np.argmax(res)])
                #                 accuracy.append(str(res[np.argmax(res)]*100))
                #         else:
                #             sentence.append(actions[np.argmax(res)])
                #             accuracy.append(str(res[np.argmax(res)]*100)) 

            if len(sentence) > 1: 
                    sentence = sentence[-1:]
                    accuracy=accuracy[-1:]

                # Viz probabilities
                # frame = prob_viz(res, actions, frame, colors,threshold)
            
        else:
            if currtent_time - last_hand_time > timeout:
                sequence = []
                predictions.clear()
        cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame,"Output: -"+' '.join(sentence)+''.join(accuracy), (10,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        frame[40:400, 0:300] = cropframe
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()