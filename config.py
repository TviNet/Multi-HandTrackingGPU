WINDOW = "Hand Tracking"
mode = "tflite"
#mode = "pb"
if mode == "tflite":
    PALM_MODEL_PATH = "./palm_detection_without_custom_op.tflite"
    LANDMARK_MODEL_PATH = "./hand_landmark_small.tflite"
else:
    # PALM_MODEL_PATH = "./palm_detection_without_custom_op.pb"
    # LANDMARK_MODEL_PATH = "./hand_landmark.pb"
    PALM_MODEL_PATH = "./saved_model_hand_landmark/saved_model.pb"
    LANDMARK_MODEL_PATH = "./saved_model_palm_detection_builtin/saved_model.pb"
ANCHORS_PATH = "./anchors.csv"
VIDEO = 0
#VIDEO = 'http://192.168.29.172:4747/mjpegfeed'
#VIDEO = 'http://192.168.29.172:8080/video'
GET_IPWEBCAM = False
#IMAGE = 'http://192.168.29.172:8080/shot.jpg'

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

FILTER_COEFFS = [-0.17857, -0.07143, 0.03571, 0.14286, 0.25, 0.35714, 0.46429]
#FILTER_COEFFS = [-0.12088, -0.08791, -0.05495, -0.02198, 0.01099, 0.04396, 0.07692, 0.10989, 0.14286, 0.17582, 0.20879, 0.24176, 0.27473]
NATIVE_RES_X = 1920
NATIVE_RES_Y = 1080
WEBCAM_X = 640
WEBCAM_Y = 480
CAM_RES_X = WEBCAM_X // 2#320#640 
CAM_RES_Y = WEBCAM_Y // 2 #240#480
SCALE = 1
FLIP_X = False
FLIP_Y = False

LOG = False

SAVE_FILE = "DATASET/train.pkl"
TRAIN_GESTURE = "CLICK"
LABELS = ["RELEASE", "HOLD", "CLICK", "ROTATE", "POINT"]
MODEL_TYPE = 'svm'
SKIP_THUMB = True