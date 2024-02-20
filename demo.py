import time

from PyQt5.QtWidgets import QApplication

from objdetector import Detector
from objtracker import *
import imutils
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# VIDEO_PATH = './video/videotest.avi'
VIDEO_PATH = r"C:\Users\Administrator\Desktop\shangaocamera1.mp4"
# VIDEO_PATH = "rtsp://admin:haivision123@192.168.1.18:554/Streaming/Channels/101"
RESULT_PATH = 'demo.mp4'

def main():

    name = 'demo'
    det = Detector()
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(5)
    print('fps:', fps)
    t = int(1000/fps)
    size = None
    videoWriter = None
    isVideoWriter = True #是否保存视频

    while True:

        # try:
        _, im = cap.read()
        frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if(frames % 2 == 0):
            continue
        if im is None:
            print("error-读取视频流失败！")
            break
        t1 = time.time()
        result = det.feedCap(im)
        t2 = time.time()
        result = result['frame']
        result = imutils.resize(result, height=500)
        # result = im.copy()
        if isVideoWriter:
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))
            videoWriter.write(result)
        t = t2 - t1
        # print("处理时间", t)

        cv2.imshow(name, result)
        cv2.waitKey(1)

    #     if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
    #         # 点x退出
    #         break
    #
    # cap.release()
    # videoWriter.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()