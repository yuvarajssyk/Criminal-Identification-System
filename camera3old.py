# camera.py
import numpy as np
import os
import cv2
import PIL.Image
from PIL import Image
import shutil
class VideoCamera3(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        #self.video = cv2.VideoCapture(0)
        self.k=1

        ff=open("video.txt","r")
        fn=ff.read()
        ff.close()
        self.filename = "video.avi"
        frames_per_second = 24.0
        res = '720p'

        # Set resolution for the video capture
        # Function adapted from https://kirr.co/0l6qmh
        def change_res(cap, width, height):
            cap.set(3, width)
            cap.set(4, height)

        # Standard Video Dimensions Sizes
        STD_DIMENSIONS =  {
            "480p": (640, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        }


        # grab resolution dimensions and set video capture to it.
        def get_dims(cap, res='1080p'):
            width, height = STD_DIMENSIONS["480p"]
            if res in STD_DIMENSIONS:
                width,height = STD_DIMENSIONS[res]
            ## change the current caputre device
            ## to the resulting resolution
            change_res(cap, width, height)
            return width, height

        # Video Encoding, might require additional installs
        # Types of Codes: http://www.fourcc.org/codecs.php
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        def get_video_type(filename):
            filename, ext = os.path.splitext(filename)
            if ext in VIDEO_TYPE:
              return  VIDEO_TYPE[ext]
            return VIDEO_TYPE['avi']



        self.cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter(self.filename, get_video_type(self.filename), 25, get_dims(self.cap, res))

        
    
    def __del__(self):
        self.cap.release()
        
    
    def get_frame(self):
        success, image = self.cap.read()
        
            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
