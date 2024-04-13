import mediapipe as mp
import cv2
import numpy as np
import time




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



# cap2 = cv2.VideoCapture(1)

def track_movement(cap, width, height):
    count = 0
    # stage = None
    start_time = time.time() 
    has_detected = False
    timestamp = None
    
    final_count = 0
    
    out_filename = f'output_pose_estimation.mp4'
    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            count += 1
            ret, frame = cap.read()
            
            try:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                break
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            try:
                # get coordinates
                
                landmarks = results.pose_landmarks.landmark
                LEFTshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                NOSEcoords = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value].x

                if time.time() - start_time < 3 :
                    orginal= nose
                    print("Starting line location: ", nose)
                #print(nose)

            # ON SCREEN COORDINATES --> Nose
                
                cv2.putText(image, str(nose),
                            tuple(np.multiply(NOSEcoords, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255),2, cv2.LINE_AA)

            # checks to see if right shoulder passes starting line coordinate
                if nose < orginal - .03:
                    status = "DETECTION"
                    false_start = True
                    if has_detected == False:
                        detection_coords = nose
                        timestamp = time.time() - start_time
                        print ("Detection time: ", timestamp)
                        print("Detection coordinaters: ", detection_coords)
                        has_detected = True
                        stop_time = timestamp + 2
                        final_count = count
                else: 
                    status = "READY"
                    false_start = False
                if stop_time <= time.time()-start_time :
                    cv2.destroyAllWindows()
                    return timestamp, count
            except:
                pass

            #on screen text
            if false_start == False:
                cv2.rectangle(image, (0,0), (190, 50), (255,0,0), -1)
            if false_start == True:
                cv2.rectangle(image, (0,0), (190, 50), (0,0,255), -1)
            if has_detected == True: 
                    cv2.rectangle(image, (0,0), (190, 50), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (600, 100), (0,0,255), -1)
                    cv2.putText(image, str(status), (10,30), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(timestamp), (270, 70), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image,"Detection Time: " , (00, 70), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2, cv2.LINE_AA)


            cv2.putText(image, str(status), (10,30), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius = 2),
                                    mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius = 2)                       
                                                            )

            cv2.imshow('mediapipe frame', image)
            
            out.write(image)

            # How to find a landmark for a specific joint: -->
            # landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

        return timestamp, count