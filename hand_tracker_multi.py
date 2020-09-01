import csv
import cv2
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from nms import non_max_suppression_fast

class HandTracker():
    r"""
    Class to use Google's Mediapipe HandTracking pipeline from Python.
    So far only detection of a single hand is supported.
    Any image size and aspect ratio supported.
    """

    def __init__(self, palm_model, joint_model, anchors_path):
        self.sess_palm = tf.Session(graph=self.load_pb("./palm_detection_builtin.pb"))
        self.sess_hand = tf.Session(graph=self.load_pb("./hand_landmark_small.pb"))
        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        
        self.left_hand = {"joints":None, "palm":None, "box":None}
        self.right_hand = {"joints":None, "palm":None, "box":None}

        self.calculate_palm = True


    def load_pb(self, path_to_pb):
        with tf.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph
    
    @staticmethod
    def _im_normalize(img):
         return np.ascontiguousarray(
             2 * ((img / 255) - 0.5
        ).astype('float32'))
       
    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )
    
    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')
    
    
    def predict_joints(self, img_norm):
        joints, flag, handedness = self.sess_hand.run(['ld_21_3d:0','output_handflag:0','output_handedness:0'], feed_dict={'input_1:0':img_norm.reshape(1,256,256,3)})
        joints = joints.reshape(-1,3)
        if handedness[0,0] < 0.5:
            hand = "right"
            # joints = self.sess_hand.run(['ld_21_3d:0'], feed_dict={'input_1:0':cv2.flip(img_norm, 1).reshape(1,256,256,3)})
            # joints = joints[0].reshape(-1,3)
            # joints[:,0] = 256-joints[:,0]
        else:
            hand = "left"
        return joints, flag, hand

    
    def detect_hand(self, img_norm):
        assert -1 <= img_norm.min() and img_norm.max() <= 1,\
        "img_norm should be in range [-1, 1]"
        assert img_norm.shape == (256, 256, 3),\
        "img_norm shape must be (256, 256, 3)"

        out_reg, out_clf = self.sess_palm.run(['regressors:0', 'classificators:0'], feed_dict={'input:0':img_norm.reshape(1,256,256,3)})
        out_reg = out_reg[0]
        out_clf = out_clf[0,:,0]

        # finding the best prediction
        out_clf = np.clip(out_clf, -20, 20)
        probabilities = self._sigm(out_clf)
        detecion_mask = probabilities > 0.3
        candidate_detect = out_reg[detecion_mask]
        candidate_anchors = self.anchors[detecion_mask]
        probabilities = probabilities[detecion_mask]

        if candidate_detect.shape[0] == 0:
            print("No hands found")
            return None, None

        # Pick the best bounding box with non maximum suppression
        # the boxes must be moved by the corresponding anchor first
        moved_candidate_detect = candidate_detect.copy()
        moved_candidate_detect[:, :2] = candidate_detect[:, :2] + (candidate_anchors[:, :2] * 256)
        box_ids = non_max_suppression_fast(moved_candidate_detect[:, :4], probabilities)

        keypoints_list = []
        side_list = []

        for max_idx in box_ids:
            # bounding box offsets, width and height
            dx,dy,w,h = candidate_detect[max_idx, :4]
            center_wo_offst = candidate_anchors[max_idx,:2] * 256
            
            # 7 initial keypoints
            keypoints = center_wo_offst + candidate_detect[max_idx,4:].reshape(-1,2)
            side = max(w,h) * self.box_enlarge

            keypoints_list.append(keypoints)
            side_list.append(side)
        

        return keypoints_list, side_list
        #return keypoints, side

    def preprocess_img(self, img):
        # fit the image into a 256x256 square
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(
            img,
            ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (256, 256))
        img_small = np.ascontiguousarray(img_small)
        
        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad

    def get_cropped_image(self, img, box, angle):
        # x1, y1, x2, y2
        # if abs(angle) > 90:
        #     #self.box_shift *= -1
        #     self.box_enlarge *= 1.1
        centre = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        box_points = np.array([[box[0], box[3]],
                            [box[0], box[1]],
                            [box[2], box[1]],
                            [box[2], box[3]]], dtype="float32")
        rot_M = cv2.getRotationMatrix2D(centre, angle, self.box_enlarge)
        rotated_box = cv2.transform(np.array([box_points]), rot_M)[0]

        width = (box[2]-box[0])

        x_shift = +np.sin(np.deg2rad(angle)) * width * self.box_shift
        y_shift = +np.cos(np.deg2rad(angle)) * width * self.box_shift

        trans_M = np.float32([ [1,0,x_shift], [0,1,y_shift] ])
        # print(rotated_box.shape)
        # print(trans_M.shape)
        rot_trans_box = cv2.transform(np.array([rotated_box]), trans_M)[0]

        new_width = int(width * self.box_enlarge)
        src_pts = rot_trans_box.astype("float32")

        dst_pts = np.array([[0, 255],
                            [0, 0],
                            [255, 0],
                            [255, 255]], dtype="float32")

        # the perspective transformation matrix
        crop_M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        cropped_image = cv2.warpPerspective(img, crop_M, (256, 256))
        return cropped_image, [src_pts, dst_pts], rot_trans_box

    def inv_crop_transform(self, trans_pts, crop_keypoints):
        src_pts = trans_pts[1]
        dst_pts = trans_pts[0]
        trans_M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        #print(trans_M.shape)
        img_keypoints = cv2.transform(np.array([crop_keypoints]), trans_M)[0]

        #print(img_keypoints.shape)
        return img_keypoints[:,:2]

    def get_box_rotation(self, keypoints, shape):
        if self.calculate_palm:
            middle_direction = (keypoints[2]-keypoints[0])
        else:
            middle_direction = (((keypoints[5]+keypoints[13])*0.5+keypoints[9])*0.5-keypoints[0])

        angle = np.arctan2(-middle_direction[1], middle_direction[0]) * 180 / np.pi - 90

        if self.calculate_palm:
            palm_idxs = [i for i in range(7)]
        else:
            palm_idxs = [0,1,2,3,5,6,9,10,13,14,17,18] # [i for i in range(21)]#
            #palm_idxs = [0,1,2,3,5,6,9,10,13,14,17,18]

        xmax, xmin = keypoints[palm_idxs,0].max(), keypoints[palm_idxs,0].min() 
        ymax, ymin = keypoints[palm_idxs,1].max(), keypoints[palm_idxs,1].min() 

        centre = ((xmax+xmin)/2, (ymax+ymin)/2)
        side = max((xmax-xmin), (ymax-ymin)) 

        # side = max(side, 240/640 *256 / self.box_enlarge)
        # side = min(side, 300/640 *256 / self.box_enlarge)
        # side = max(side, 120/640 *256 / self.box_enlarge)
        # side = min(side, 240/640 *256 / self.box_enlarge)

        self.side_length = side
        self.angle = angle
        x1,y1 = max(int(centre[0]-side/2),0), max(int(centre[1]-side/2),0)
        x2,y2 = min(int(centre[0]+side/2),shape[1]), min(int(centre[1]+side/2),shape[0])

        box = [x1, y1, x2, y2]

        return box, angle, keypoints[palm_idxs]


    def __call__(self, img):
        
        self.left_hand = {"joints":None, "palm":None, "box":None}
        self.right_hand = {"joints":None, "palm":None, "box":None}

        img_pad, img_norm, pad = self.preprocess_img(img)
        
        self.box_enlarge = 2.6
        self.box_shift = -0.5
        
        palm_keypoints_list, side_list = self.detect_hand(img_norm)
        #palm_keypoints, side = self.detect_hand(img_norm)
        if palm_keypoints_list is None:
            return None, None, None

        for palm_keypoints in palm_keypoints_list:

            box, angle, kp_considered = self.get_box_rotation(palm_keypoints, img_norm.shape)
            cropped, trans_pts, box_considered = self.get_cropped_image(img_norm, box, angle)
            img_landmark = cropped #hand_square

            # calculate joints
            joints_3d, flag, handedness = self.predict_joints(img_landmark)
            joints = joints_3d.copy()[:,:2]
            rotated_joints = self.inv_crop_transform(trans_pts, joints)
            
            kp_orig = rotated_joints
            kp_palm = kp_considered 
            box = box_considered.astype(float)
            box *= max(img.shape[0], img.shape[1])/256
            kp_orig *= max(img.shape[0], img.shape[1])/256
            kp_palm *= max(img.shape[0], img.shape[1])/256
           
            kp_orig -= pad[::-1]
            box -= pad[::-1]
            kp_palm -= pad[::-1]
            
            if handedness == "left":
                self.left_hand["joints"] = joints_3d
                self.left_hand["joints"][:,:2] = kp_orig
                self.left_hand["box"] = box
                self.left_hand["palm"] = kp_palm
                self.left_hand["angle"] = angle

            else:
                self.right_hand["joints"] = joints_3d
                self.right_hand["joints"][:,:2] = kp_orig
                self.right_hand["box"] = box
                self.right_hand["palm"] = kp_palm
                self.right_hand["angle"] = angle

            # self.joint_3d_coords[:,:2] = kp_orig
            # self.box_hand = box

            # if kp_palm.shape[0] == 7:
            #     self.palm_keypoints = kp_palm
            # else:
            #     self.palm_keypoints = kp_palm[[0,1,4,6,8,10],:]