import cv2
import numpy as np
from PIL import Image

from hand_tracker_multi import HandTracker
from process_keypoints import *
from gesture import Gestures
from trainer import TrainData, Model
from control import Control
from drawing_helpers import *
import time
from config import *
import os
import requests

cv2.namedWindow(WINDOW)

if not GET_IPWEBCAM:
    capture = cv2.VideoCapture(VIDEO)
    result = cv2.VideoWriter('output.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, (640, 480)) 
    images = []
    # capture.set(cv2.CAP_PROP_AUTO_EXPOSURE,0)
    # capture.set(cv2.CAP_PROP_EXPOSURE,-6)

    if capture.isOpened():
        hasFrame, frame = capture.read()
    else:
        hasFrame = False
else:
    img_req = requests.get(IMAGE)
    img_arr = np.array(bytearray(img_req.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    hasFrame =True

print(frame.shape)
detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
)

#gesture = Gestures()
#control = Control()
#control.update_position([NATIVE_RES_X//2 , NATIVE_RES_Y//2])
#trainer = TrainData()
#model =  Model(MODEL_TYPE)
#trainer.read_data(SAVE_FILE)

def hist_eq(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4)
    lab[...,0] = clahe.apply(lab[...,0])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame

def gamma_correc(frame, gamma = 1.0):
    invGamma = 1 / gamma 
    table = np.array([ ( (i/255)**invGamma ) * 255 for i in range(0,256)]).astype(np.uint8)
    return cv2.LUT(frame, table)

def process_hand(frame, hand_data, hand_name, hand_history):

    box = hand_data["box"]
    palm_points = hand_data["palm"]
    angle = hand_data["angle"]
    points = hand_data["joints"][:,:2]

    if box is not None:
        draw_box(frame, box)

    if palm_points is not None:
        pass
        #draw_keypoints(frame, palm_points)

    if points is not None:

        rot_keypoints, norm_keypoints = normalize_keypoints(hand_data["joints"])
        orientation = orientation_keypoints(hand_data["joints"])

        #control.update_position(detector.joint_3d_coords[8,:2].tolist())
        if hand_name == "left":
            box_centre = hand_data["joints"][5,:2]#detector.palm_keypoints.astype(float).mean(0)
            #box_centre = np.array(box.mean(axis=0)).astype(float)

            angle = angle - 10 #orientation - 90 - 10
            x_shift = 0#+np.sin(np.deg2rad(angle)) * 30 * -2
            y_shift = 0#+np.cos(np.deg2rad(angle)) * 30 * -2
            pointing = box_centre + np.array([x_shift, y_shift])

            hand_history.append(pointing)

            if len(hand_history) > len(FILTER_COEFFS):
                #hand_history = hand_history[1:]
                hand_history.pop(0)
                smoothed_keypoints = smooth_keypoints(hand_history)
            else:
                smoothed_keypoints = hand_history[-1]
            #control.update_position(smoothed_keypoints)
            # cv2.circle(frame, (int(smoothed_keypoints[0]), int(smoothed_keypoints[1])), 10, (255, 255, 255), -1)
        else:
            # ges, ges_desc, ges_ang = gesture.estimate_gesture(rot_keypoints)
            # ges_predict = model.predict([ges_ang])
            # ges_avg = gesture.current_detected_gesture(ges_predict)
            # #control.command(ges_avg)
            # # draw_text(frame, ges_predict, ges_avg)
            # gesture.empty()
            pass

        draw_keypoints(frame, points)
        draw_connections(frame, points)

        if LOG:
            print(pos, end=", ")
            print(ang, end=", ")
            print(ges_ang * 180 / np.pi, end=", ")

left_history = []
right_history = []
# for key in trainer.train_data.keys():
#     print(f"{key}-{len(trainer.train_data[key])}")

while hasFrame:
    frame = cv2.flip(frame, 1)
    #frame = gamma_correc(frame, 2)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.rectangle(frame, ((WEBCAM_X-CAM_RES_X)//2, (WEBCAM_Y-CAM_RES_Y)//2), (WEBCAM_X - (WEBCAM_X-CAM_RES_X)//2, WEBCAM_Y - (WEBCAM_Y-CAM_RES_Y)//2), (0, 0, 255), 2)
    ts = time.time()
    
    detector(image)

    frame_time = time.time()-ts
    frame_rate = 1 / frame_time

    if LOG:
        print(detector.calculate_palm, end=", ")
        print(round(frame_time, 4), end=", ")

    if detector.left_hand["joints"] is not None:
        process_hand(frame, detector.left_hand, "left", left_history)
    else:
        left_history = []

    #print(len(right_history))
    if detector.right_hand["joints"] is not None:
        process_hand(frame, detector.right_hand, "right", right_history)
    else:
        right_history = []

    cv2.imshow(WINDOW, frame)
    #result.write(frame)
    #images.append(Image.fromarray(frame[:,:,::-1]))

    if not GET_IPWEBCAM:
        hasFrame, frame = capture.read()
    else:
        img_req = requests.get(IMAGE)
        img_arr = np.array(bytearray(img_req.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        hasFrame = True

    total_time = time.time()-ts
    total_rate = 1 / total_time
    if LOG:
        print(round(total_time, 4), end=", ")
        print()

    key = cv2.waitKey(1)
    if key == ord('s'):
        #trainer.add_data(TRAIN_GESTURE, ges_ang)
        pass
    elif key == 32:
        pass
    elif key == ord('d'):
        #trainer.delete_data(TRAIN_GESTURE)
        pass
    elif key == 27:
        break

# images[0].save('out.gif',
#                save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
#trainer.save_data(SAVE_FILE)
capture.release()
cv2.destroyAllWindows()
