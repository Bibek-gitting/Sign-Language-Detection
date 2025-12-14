from SignLanguageDetectionUsingML.features import *
from time import sleep

# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    
    # NEW LOOP
    # Loop through actions
    # for action in actions:
        # Count how many frames exist in Image/<action>/
        image_path = 'Image/H'
        if not os.path.exists(image_path):
            print(f"⚠️  Skipping - Directory not found: {image_path}")
            
        frame_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])
        
        if len(frame_files) < no_sequences:
            print(f"⚠️  Warning: H has only {len(frame_files)} images, need {no_sequences}")

        # for frame_file in frame_files:
        #     frame = cv2.imread(f'Image/{action}/{frame_file}')

        # Loop through sequences aka videos
        for sequence in range(min(len(frame_files), no_sequences)):
            frame = cv2.imread(os.path.join(image_path, frame_files[sequence]))

            if frame is None:
                print(f"❌ Missing: Image/H/{sequence+1}.png")
                continue
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                # ret, frame = cap.read()
                # frame=cv2.imread('Image/{}/{}.png'.format(action,sequence))
                # frame=cv2.imread('{}{}.png'.format(action,sequence))
                # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                # Enhance contrast/brightness
                # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # v = hsv[:, :, 2]
                # v = cv2.equalizeHist(v)
                # hsv[:, :, 2] = v
                # frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # Make detections
                image, results = mediapipe_detection(frame, hands)
    #                 print(results)
                if results.multi_hand_landmarks is None:
                    # Slightly blur or adjust contrast and retry
                    frame = cv2.GaussianBlur(frame, (3,3), 0)
                    image, results = mediapipe_detection(frame, hands)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format("H", sequence+1), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format("H", sequence+1), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                if keypoints is not None and np.sum(keypoints) != 0:
                    npy_path = os.path.join(DATA_PATH, "H", str(sequence+1), f'{frame_num+1}.npy')
                    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                    np.save(npy_path, keypoints)
                else:
                    print(f"⚠️  No hand detected in H image {frame_files[sequence]}")

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        # cap.release()
        cv2.destroyAllWindows()


