from config import *
import cv2
import numpy as np

def draw_box(frame, box):
    box = box.astype(int)
    cv2.circle(frame, (int(box.mean(axis=0)[0]), int(box.mean(axis=0)[1])), 1 * 2, (0, 0, 255), 2)
    cv2.drawContours(frame, [box.astype(int)], 0, (0, 0, 255), 2)

def draw_keypoints(frame, keypoints):
    for point in keypoints:
        x, y = point
        cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)

def draw_connections(frame, points):
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

def draw_text(frame, ges_predict, ges_avg):
    cv2.putText(frame,LABELS[ges_predict[0]],(400,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
    cv2.putText(frame,ges_avg,(400,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)