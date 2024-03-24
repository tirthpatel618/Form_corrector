# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time


# Initialize video capture
def start_capture(path):
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    return cap, count, fps, height, width

#releasing the capture
def release(capture):
    capture.release()

#plotting the frames of the video
def plot(capture, frame_count):
    fig, axs = plt.subplots(6, 6, figsize=(30, 30))
    axs = axs.flatten()
    img_idx = 0
    for frame in range(frame_count):
        ret, img = capture.read()
        if ret == False:
            break
        if frame % 10 == 0:
            axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[img_idx].set_title(f'Frame: {frame}')
            axs[img_idx].axis('off')
            img_idx += 1

    plt.tight_layout()
    plt.show()



ballLower = (25,41,98)
ballUpper = (32,78,92)


def ball_tracking(cap, count, fps, height, width):
    coords = []
    pts = deque(maxlen=64)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ballLower, ballUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            coords.append([x, y])
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        cv2.imshow("Frame", frame)
    cap.release()
    cv2.destroyAllWindows()
    return coords



def main():
    cap, count, fps, height, width = start_capture('/Users/tirthpatel/Desktop/Code/project3/free_throw_made.mp4')
    coords = ball_tracking(cap, count, fps, height, width)
    print(coords)
main()