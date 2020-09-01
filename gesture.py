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
import numpy as np
from collections import Counter
from config import *
from gesture_rules import GestureRules

class Gestures:
	def __init__(self):
		
		self.cur_gesture = None
		self.num_hands = 1

		self.enum = {"THUMB":0, "INDEX":1, "MIDDLE":2, "RING":3, "LITTLE":4}
		self.inv_enum =  {v: k for k, v in self.enum.items()} 

		self.pointing_finger = "MIDDLE"
		self.predefined_gestures = ["CLICK", "HOLD", "RELEASE", "ROTATE"]

		self.position = None
		self.orientation = None
		self.gesture = None
		self.gesture_rules = GestureRules()

		self.NUM_GESTURES_TO_STORE = 5
		self.GESTURE_THRESHOLD = 0.8
		self.gesture_list = []

	def add_to_list(self, gesture):
		self.gesture_list.append(gesture)
		if len(self.gesture_list) > self.NUM_GESTURES_TO_STORE:
			self.gesture_list = self.gesture_list[1:]

	def current_detected_gesture(self, ges_predict): 
		self.add_to_list(LABELS[ges_predict[0]])
		counts = Counter(self.gesture_list)
		for key, val in counts.items():
			if val >= self.GESTURE_THRESHOLD * self.NUM_GESTURES_TO_STORE:
				return key
		return "None"

	def empty(self):
		self.position = None
		self.orientation = None
		self.gesture = None
		self.gesture_rules.empty()

	def set_position(self, keypoints):
		self.position = keypoints[self.enum[self.pointing_finger]*4 + 3 + 1, :2]
		return self.position

	def set_orientation(self, angle):
		self.orientation = angle
		return self.orientation

	def theta(self, v, w): 
		ang = v.dot(w)/(np.linalg.norm(v)*np.linalg.norm(w))
		return np.arccos(np.clip(ang, -1, 1))


	def calculate_angles(self, keypoints):
		angles = np.zeros((5, 3))
		mcps = [1,5,9,13,17]
		pips = [2,6,10,14,18]
		dips =  [3,7,11,15,19]
		tips = [4,8,12,16,20]
		v0 = keypoints[tips]-keypoints[dips]
		v1 = keypoints[dips]-keypoints[pips]
		v2 = keypoints[pips]-keypoints[mcps]
		v3 = keypoints[mcps]

		for i in range(5):
			angles[i, 0] = self.theta(v3[i], v2[i])  
			angles[i, 1] = self.theta(v2[i], v1[i]) 
			angles[i, 2] = self.theta(v1[i], v0[i]) 

		return angles

	def estimate_gesture(self, keypoints):
		angles = self.calculate_angles(keypoints)
		self.gesture = None#self.gesture_rules.get_matching_gesture(angles)
		
		return self.gesture, self.gesture_rules.gesture_read, angles

