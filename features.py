#import dependency
import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results):
    if not results.multi_hand_landmarks:
      return None
    hand_landmarks = results.multi_hand_landmarks[0]
    handedness = results.multi_handedness[0].classification[0].label

    #Extract raw keypoints
    kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    #Flip LEFT hand → RIGHT hand
    if handedness == "Left":
        kp[:, 0] =1.0 - kp[:, 0]
    
    #Translate (make wrist origin)
    wrist = kp[0].copy()
    kp[:, :3] -= wrist

    #Scale normalization (size invariance)
    ref_dist = np.linalg.norm(kp[5]) + 1e-6
    kp[:, :3] /= ref_dist

    return kp.flatten()
      # for hand_landmarks in results.multi_hand_landmarks:
      #   rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
      #   return(np.concatenate([rh]))

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

no_sequences = 200
sequence_length = 1

