import time

from objdetector import Detector
import imutils
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# VIDEO_PATH = './video/videotest.avi'
VIDEO_PATH = "rtsp://admin:Liuwei123456@192.168.1.20:554/stream1"
RESULT_PATH = 'demo.mp4'

def main():

    func_status = {}
    func_status['headpose'] = None
    
    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(5)
    print('fps:', fps)
    t = int(1000/fps)
    size = None
    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        t1 = time.time()
        result = det.feedCap(im, func_status)
        t2 = time.time()
        result = result['frame']
        result = imutils.resize(result, height=500)
        # result = im.copy()
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))
        t = t2 - t1
        print("处理时间", t)
        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(1)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()