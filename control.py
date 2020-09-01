import win32api, win32con
from config import *
import pyautogui, sys

pyautogui.FAILSAFE = False

class Control:
    def __init__(self):
        self.W = NATIVE_RES_X
        self.H = NATIVE_RES_Y
        self.camW = CAM_RES_X
        self.camH = CAM_RES_Y
        self.flipx = FLIP_X
        self.flipy = FLIP_Y
        self.scale = SCALE

        self.state = "START"
        self.prev_gesture = "None"
        self.STATE_MACHINE = {
        "CLICK" : {"START":"move", "move":"move", "RELEASE":"END"},
        "HOLD" : {"START":"move", "move":"move", "RELEASE":"END"}
        }

    def command(self, gesture):

        if gesture == "CLICK":
            self.click_down(self.position)
        elif gesture == "RELEASE":
            self.click_up(self.position)
        elif gesture == "POINT":
            self.scroll(10)
        else:
            pass

    def update_position(self, input_position):
        position = [input_position[0], input_position[1]]
        position[0] = min(position[0] - (640 - self.camW) // 2, self.camW)
        position[0] = min(position[0] * self.W / self.camW , self.W) 
        if self.flipx:
            position[0] = self.W - position[0]
        position[1] = min(position[1] - (480 - self.camH) // 2, self.camH)
        position[1] = min(position[1] * self.H / self.camH , self.H) 
        if self.flipy:
            position[1] = self.H - position[1]
        self.position = [int(position[0]), int(position[1])]
        #print(self.position)
        self.move(self.position)

    def move(self, position):
        #win32api.SetCursorPos((position[0],position[1]))
        pyautogui.moveTo(position[0], position[1]) 

    def click_down(self, position):
        x,y = position[0],position[1]
        # win32api.SetCursorPos((x,y))
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        #
        pyautogui.mouseDown(button='left');
        #win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    def click_up(self, position):
        x,y = position[0],position[1]
        # win32api.SetCursorPos((x,y))
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
        pyautogui.mouseUp(button='left');

    def click(self, position):
        x,y = position[0],position[1]
        # win32api.SetCursorPos((x,y))
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
        pyautogui.click(button='left') 

    def scroll(self, amount=10):
        pyautogui.scroll(amount)