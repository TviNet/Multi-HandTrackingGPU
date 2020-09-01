# Multi-HandTrackingGPU Python
Multi-Hand Tracking on GPU using Mediapipe models in python

```
$ pip install opencv-python tensorflow
$ python run.py
```
Mediapipe currently does not have GPU support for windows. This repo provides high FPS multi-hand tracking using GPU on python. 

![Output](/out.gif?raw=true "Output")

References:<br/>
*mediapipe-models*: https://github.com/junhwanjang/mediapipe-models/tree/master/palm_detection/mediapipe_models <br/>
*mediapipe*: https://github.com/google/mediapipe/tree/master/mediapipe/models <br/>
*hand_tracking*: https://github.com/wolterlw/hand_tracking , https://github.com/metalwhale/hand_tracking <br/>
*Convert tflite to pb base*: https://gist.github.com/tworuler/bd7bd4c6cd9a8fbbeb060e7b64cfa008 <br/>
