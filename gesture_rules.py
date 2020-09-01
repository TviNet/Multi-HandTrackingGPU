import numpy as np
from collections import Counter

class GestureRules:
	def __init__(self):
		self.angle_types = ["STRAIGHT", "BENT", "RIGHT"]
		self.predefined_gestures = ["CLICK", "HOLD", "RELEASE", "ROTATE"]
		self.gesture_definition = {"CLICK": {},
									"HOLD": {},
									"RELEASE": {},
									"ROTATE": {},

									}
		self.ANGLE_THESHOLD = 20
		self.enum = {"THUMB":0, "INDEX":1, "MIDDLE":2, "RING":3, "LITTLE":4}
		self.inv_enum =  {v: k for k, v in self.enum.items()} 
		self.gesture_read = {"THUMB":[],
							"INDEX":[],
							"MIDDLE":[],
							"RING":[],
							"LITTLE":[]
							}
		self.gesture_angles = None

	def empty(self):
		self.gesture_angles = None
		self.gesture_read = {"THUMB":[],
							"INDEX":[],
							"MIDDLE":[],
							"RING":[],
							"LITTLE":[]
							}

	def angle_to_type(self, angle):
		if abs(angle * 180 / np.pi - 0) < 20:
			return "STRAIGHT"
		elif abs(angle * 180 / np.pi - 90) < 20:
			return "RIGHT"
		else:
			return "BENT"

	def read_gesture(self, angles):
		for i in range(5):
			for j in range(3):
				self.gesture_read[self.inv_enum[i]].append(self.angle_to_type(angles[i,j]))

	def compare_definition(self, definition, angles):
		for i in range(5):
			for j in range(3):
				if definition[self.inv_enum[i]][j] != "ANY" and definition[self.inv_enum[i]][j] != self.angle_to_type(angles[i,j]) :
					return False
		return True

	def get_matching_gesture(self, angles):
		self.read_gesture(angles)
		self.gesture_angles = angles
		for gesture_name, gesture_definition in self.gesture_definition.items():
			if self.compare_definition(gesture_definition, angles):
				return gesture_name
		return None
