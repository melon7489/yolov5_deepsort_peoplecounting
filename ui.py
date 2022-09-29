import sys
import time
import cv2
import numpy as np
import os
from PyQt5.QtCore import QThread, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from objtracker import Objtracker
from objdetector import Detector
from test import Ui_MainWindow

RECT_AREA_RATIO = 0.05    # 人体占整个图像最小比例阈值
ROI_LT = (520, 140)  # 感兴趣区域 左上 和 右下 坐标
ROI_RB = (1400, 1080)
WAITFRAME = 50
class LoadFrameThread(QThread):  # 加载帧
    def __init__(self, predThread):
        super(LoadFrameThread, self).__init__()
        self.predThread = predThread

    def run(self):
        cap = self.predThread.video
        while cap.isOpened():
            isvalid, frame = cap.read()
            if isvalid:
                if len(self.predThread.framelist) < self.predThread.MAX_FRAME_LIST:
                    self.predThread.mutex.lock()
                    self.predThread.framelist.append(frame)
                    self.predThread.mutex.unlock()
                else:
                    print("警告：缓冲区最大容量为:{}，缓冲区已满，可能导致性能下降！".format(self.predThread.MAX_FRAME_LIST))

class PredThread(QThread):  # 预测线程
    def __init__(self, Mainwin, videotype):
        super(PredThread, self).__init__()
        self.mainwin = Mainwin
        self.videotype = videotype
        if self.videotype == "back":
            self.video = self.mainwin.srcVideo_back
            self.filename = self.mainwin.filename_back
        else:
            self.video = self.mainwin.srcVideo
            self.filename = self.mainwin.filename
        self.objtracker = Objtracker()
        self.detector = Detector()
        # list 与蓝色polygon重叠
        self.list_overlapping_blue_polygon = []

        # list 与黄色polygon重叠
        self.list_overlapping_yellow_polygon = []
        self.init()

        self.timer = QTimer()
        self.timer.timeout.connect(self.startVideo)
        self.framelist = []
        self.MAX_FRAME_LIST = 1000  # 视频缓冲池大小
        self.mutex = QMutex()  # load帧同步锁
        self.isDone = []
        self.loadFrameThread = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2()  # 混合高斯分布模型分离前景和背景
        self.waitFrame = 0


    def loaddata(self):
        frame = self.framelist[0]
        self.framelist.pop(0)
        return True, frame

    def init(self):  # 初始化mask及其他辅助图像  前门和后门摄像头共用一个mask
        # 打开视频,前门打开用前门，前门没打开用后门的
        originalSize = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        ## 保存视频
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 保存格式为MP4
        self.output = cv2.VideoWriter("./result/111.MP4", fourcc, self.video.get(5), originalSize)

        # 根据视频尺寸，填充供撞线计算使用的polygon
        width = 1920
        height = 1080
        mask_image_temp = np.zeros((height, width), dtype=np.uint8)
        # 填充第一个撞线polygon（蓝色）
        list_pts_blue = [[0, 550], [0, 600], [1920, 600], [1920, 550]]
        ndarray_pts_blue = np.array(list_pts_blue, np.int32)
        polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
        polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

        # # 填充第二个撞线polygon（黄色）
        mask_image_temp = np.zeros((height, width), dtype=np.uint8)
        list_pts_yellow = [[0, 620], [0, 670], [1920, 670], [1920, 620]]
        ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
        polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
        polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

        # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
        polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

        # 缩小尺寸，1920x1080->960x540
        # polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (width//2, height//2))
        self.polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, originalSize)

        # 蓝 色盘 b,g,r
        blue_color_plate = [255, 0, 0]
        # 蓝 polygon图片
        blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)  # 二维图像*【255，0，0】-->单通道变3通道，BGR=蓝色

        # 黄 色盘
        yellow_color_plate = [0, 255, 255]
        # 黄 polygon图片
        yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

        # 彩色图片（值范围 0-255）
        color_polygons_image = blue_image + yellow_image

        # 缩小尺寸，1920x1080->960x540
        # color_polygons_image = cv2.resize(color_polygons_image, (width//2, height//2))
        self.color_polygons_image = cv2.resize(color_polygons_image, originalSize)

    def startVideo(self):  # 视频开始
        # if self.video.isOpened() & self.timer.isActive():
        # while self.video.isOpened():
        # 若为文件，不是网络摄像头流
        if os.path.isfile(self.filename):
            if self.video.isOpened():
                retval, im = self.video.read()
                if retval == True:
                    if self.videotype == "back":
                        ## 展示原视频
                        self.mainwin.showVideo("back", "src", im)
                    else:
                        ## 展示原视频
                        self.mainwin.showVideo("front", "src", im)
                self.process(im)
            if self.video.get(1) == self.video.get(7):  # 播放到最后一帧
                if self.videotype == "back":
                    self.mainwin.srcVideo_back = None
                else:
                    self.mainwin.srcVideo = None
                self.mainwin.clearThread(self)
                QMessageBox.information(self.mainwin, "提示！", "检测结束！", QMessageBox.Yes)
        else:
            self.loadFrameThread = LoadFrameThread(self)
            self.loadFrameThread.start()
            while self.loadFrameThread.isRunning():
                if len(self.framelist) != 0:
                    print(len(self.framelist))
                    self.mutex.lock()
                    retval, im = self.loaddata()
                    self.mutex.unlock()
                    if retval == True:
                        if self.videotype == "back":
                            ## 展示原视频
                            self.mainwin.showVideo("back", "src", im)
                        else:
                            ## 展示原视频
                            self.mainwin.showVideo("front", "src", im)
                    if self.waitFrame == 0:
                        if not self.isPeople(im):
                            # 没有目标可以抽帧
                            if len(self.framelist) > 3:
                                for i in range(3):
                                    self.framelist.pop(0)
                            output_image_frame = cv2.add(im, self.color_polygons_image)
                            if self.videotype == "back":
                                ## 展示结果
                                self.mainwin.showVideo("back", "det", output_image_frame)
                            else:
                                ## 展示结果
                                self.mainwin.showVideo("front", "det", output_image_frame)
                        else:
                            self.waitFrame = 1
                            self.process(im)
                    else:
                        self.waitFrame += 1
                        if self.waitFrame == WAITFRAME:
                            self.waitFrame = 0
                        self.process(im)

    def isPeople(self, frame):
        mask = self.fgbg.apply(frame)  # 分离前景和背景
        mask = cv2.medianBlur(mask, 5)  # 中值滤波
        mask = cv2.erode(mask, (3, 3))  # 腐蚀
        retval, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # 二值化
        roi = mask[ROI_LT[1]:ROI_RB[1], ROI_LT[0]:ROI_RB[0]]
        greyScale_map = cv2.calcHist([roi], [0], None, [256], [0, 256])  # 统计灰度直方图
        radio = greyScale_map[255][0] / ((ROI_RB[1] - ROI_LT[1]) * (ROI_RB[0] - ROI_LT[0]))  # 计算移动物体占比
        print('radio:', radio)
        cv2.imshow("", roi)
        cv2.waitKey(1)
        if radio > RECT_AREA_RATIO:
            return True
        else:
            return False

    def process(self, im):
        ## 展示处理结果
        t1 = time.time()
        list_bboxs = []
        # 更新跟踪器
        output_image_frame, list_bboxs = self.objtracker.update(self.detector, im)
        # 输出图片
        output_image_frame = cv2.add(output_image_frame, self.color_polygons_image)
        # output_image_frame = cv2.resize(output_image_frame, (1920, 1080))

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, _, track_id = item_bbox
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                # 撞线的点
                y = y1_offset
                x = x1
                if self.polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in self.list_overlapping_blue_polygon:
                        self.list_overlapping_blue_polygon.append(track_id)
                    if track_id in self.list_overlapping_yellow_polygon:
                        # 判断 黄polygon list里是否有此 track_id
                        # 有此track_id，则认为是下行方向
                        if track_id not in self.isDone: # 如果没有处理
                            # 下行+1
                            self.mainwin.mutex.lock()
                            self.mainwin.down_count += 1
                            self.mainwin.mutex.unlock()
                            # 存入已处理
                            self.isDone.append(track_id)
                            print('down count:', self.mainwin.down_count, ', down id:',
                                  self.list_overlapping_yellow_polygon)
                            # 删除 黄polygon list 中的此id
                            self.list_overlapping_yellow_polygon.remove(track_id)
                        else:
                            self.mainwin.mutex.lock()
                            if self.mainwin.up_count > 0:
                                self.mainwin.up_count -= 1
                            self.mainwin.mutex.unlock()
                            self.list_overlapping_yellow_polygon.remove(track_id)
                            self.isDone.remove(track_id)

                elif self.polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in self.list_overlapping_yellow_polygon:
                        self.list_overlapping_yellow_polygon.append(track_id)
                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则认为是上行方向
                    if track_id in self.list_overlapping_blue_polygon:
                        if track_id not in self.isDone:
                            # 上行+1
                            self.mainwin.mutex.lock()
                            self.mainwin.up_count += 1
                            self.mainwin.mutex.unlock()
                            self.isDone.append(track_id)
                            print('up count:', self.mainwin.up_count, ', up id:', self.list_overlapping_blue_polygon)
                            # 删除 蓝polygon list 中的此id
                            self.list_overlapping_blue_polygon.remove(track_id)
                        else:
                            self.mainwin.mutex.lock()
                            if self.mainwin.down_count > 0:
                                self.mainwin.down_count -= 1
                            self.mainwin.mutex.unlock()
                            self.list_overlapping_blue_polygon.remove(track_id)
                            self.isDone.remove(track_id)

            # ----------------------清除无用id（不在黄蓝线列表里的）----------------------
            list_overlapping_all = self.list_overlapping_yellow_polygon + self.list_overlapping_blue_polygon  # 列表里存的都是碰到线到的id（无论上下车与否）
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                if not is_found:
                    # 如果没找到(在黄蓝线列表外)，删除id
                    if id1 in self.list_overlapping_yellow_polygon:
                        self.list_overlapping_yellow_polygon.remove(id1)

                    if id1 in self.list_overlapping_blue_polygon:
                        self.list_overlapping_blue_polygon.remove(id1)
            list_overlapping_all.clear()
            # 清空list
            list_bboxs.clear()
        else:
            # 如果图像中没有任何的bbox，则清空list
            self.list_overlapping_blue_polygon.clear()
            self.list_overlapping_yellow_polygon.clear()
            self.isDone.clear()

        t2 = time.time()
        print("时间：", t2 - t1)
        self.mainwin.mutex.lock()
        self.mainwin.lcdNumber_in.display(self.mainwin.up_count)
        self.mainwin.lcdNumber_out.display(self.mainwin.down_count)
        # self.mainwin.lcdNumber_online.display(self.mainwin.up_count - self.mainwin.down_count if self.mainwin.up_count - self.mainwin.down_count > 0 else 0)
        self.mainwin.lcdNumber_online.display(self.mainwin.down_count - self.mainwin.up_count if self.mainwin.down_count - self.mainwin.up_count > 0 else 0)
        self.mainwin.mutex.unlock()
        self.output.write(output_image_frame)
        if self.videotype == "back":
            ## 展示结果
            self.mainwin.showVideo("back", "det", output_image_frame)
        else:
            ## 展示结果
            self.mainwin.showVideo("front", "det", output_image_frame)

    def run(self):
        if os.path.isfile(self.filename):
            self.timer.start(30)
        self.startVideo()



class Mainwin(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Mainwin, self).__init__()
        self.setupUi(self)
        # self.setFixedSize(self.width(), self.height())  # 设置窗口大小不可调
        self.open.clicked.connect(self.openfile)  # 打开文件
        self.close.clicked.connect(self.closeApp)  # 关闭程序
        self.clearNum.clicked.connect(self.clearAllNum)
        self.start.clicked.connect(self.startVideo)  # 点击开始
        self.stop.clicked.connect(self.stopVideo)  # 点击暂停
        self.open2.clicked.connect(self.openfile)  # 打开文件  后门
        self.start2.clicked.connect(self.startVideo)  # 点击开始  后门
        self.stop2.clicked.connect(self.stopVideo)  # 点击暂停  后门
        self.openWebcam_front.clicked.connect(self.openCamFront)  # 点击打开前门摄像头
        self.openWebcam_back.clicked.connect(self.openCamBack)  # 点击打开后门摄像头
        self.srcVideo = None  # 存放视频
        self.srcVideo_back = None  # 存放视频
        self.filename = None  # 文件名
        self.filename_back = None  # 文件名
        self.videoThread = None  # 前门视频线程
        self.videoThread_back = None  # 后门视频线程
        self.mutex = QMutex()  # 线程锁
        # 下行数量
        self.down_count = 0
        # 上行数量
        self.up_count = 0

    def openfile(self):  # 打开视频文件
        filename, _ = QFileDialog.getOpenFileName(self, "打开视频文件", ".", "*.*")  # 打开文件窗口，返回文件绝对路径和类型
        if filename == '':
            return
        if self.sender() == self.open:  # 打开第一组（前门）
            if self.openMedia(filename) == 0:
                QMessageBox.critical(self, "错误", "媒体未打开！", QMessageBox.Yes, QMessageBox.Yes)
                return
            self.start.setEnabled(True)
            self.stop.setEnabled(True)
            self.videoThread = PredThread(self, "front")
            self.videoThread.start()
        if self.sender() == self.open2:  # 打开第二组（后门）
            if self.srcVideo_back is not None:
                self.srcVideo_back.release()
            if self.videoThread_back is not None:
                self.clearThread(self.videoThread_back)
            if self.openMedia(filename, "back") == 0:
                QMessageBox.critical(self, "错误", "媒体未打开！", QMessageBox.Yes, QMessageBox.Yes)
                return
            self.start2.setEnabled(True)
            self.stop2.setEnabled(True)
            self.videoThread_back = PredThread(self, "back")
            self.videoThread_back.start()

    def openCamFront(self):  # 打开前门摄像头
        self.label_src.setText("loading......")
        self.label_det.setText("loading......")
        QApplication.processEvents()  # 刷新主页面
        filename = "rtsp://admin:Liuwei123456@192.168.1.247:554/Streaming/Channels/101"
        # filename = "rtsp://admin:admin123@192.168.1.102:554/cam/realmonitor?channel=1&subtype=0"
        # filename = r"G:\project\yolov5-deepsort\video\s+d.avi"
        if filename == '':
            return
        if self.openMedia(filename) == 0:
            QMessageBox.critical(self, "错误", "媒体未打开！", QMessageBox.Yes, QMessageBox.Yes)
            return
        self.start.setEnabled(False)
        self.stop.setEnabled(False)
        self.videoThread = PredThread(self, "front")
        self.videoThread.start()

    def openCamBack(self):  # 打开后门摄像头
        self.label_src2.setText("loading......")
        self.label_det2.setText("loading......")
        QApplication.processEvents()
        # filename = r"rtsp://admin:Liuwei123456@192.168.1.20:554/stream1"
        # filename = r"G:\project\yolov5-deepsort\video\s+d2.avi"
        # filename = "rtsp://admin:admin123@192.168.1.102:554/cam/realmonitor?channel=1&subtype=0"
        filename = "rtsp://admin:Liuwei123456@192.168.1.248:554/Streaming/Channels/101"
        if self.filename == '':
            return
        if self.openMedia(filename, "back") == 0:
            QMessageBox.critical(self, "错误", "媒体未打开！", QMessageBox.Yes, QMessageBox.Yes)
            return
        self.start2.setEnabled(False)
        self.stop2.setEnabled(False)
        self.videoThread_back = PredThread(self, "back")
        self.videoThread_back.start()

    def openMedia(self, filename, videotype="front"):  # 公共代码
        try:
            cap = cv2.VideoCapture(filename)
        except:
            print("流媒体未打开！")
            return 0
        if cap is None:
            return 0
        if videotype == "front":
            self.filename = filename
            if self.srcVideo is not None:
                self.srcVideo.release()
            if self.videoThread is not None:
                self.clearThread(self.videoThread)
        if videotype == "back":
            self.filename_back = filename
            if self.srcVideo_back is not None:
                self.srcVideo_back.release()
            if self.videoThread_back is not None:
                self.clearThread(self.videoThread_back)
        if cap.isOpened():
            retval, frame = cap.read()
            if retval == True:
                if videotype == "front":
                    self.showVideo("front", "src", frame)
                    self.showVideo("front", "det", frame)
                    self.srcVideo = cap
                if videotype == "back":
                    self.showVideo("back", "src", frame)
                    self.showVideo("back", "det", frame)
                    self.srcVideo_back = cap
            return 1
        else:
            return 0

    def trans(self, im, label):  # 图像转化为qt能识别的类型
        frame = im.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGBA8888)
        frame = QPixmap(frame).scaled(label.width(), label.height())
        return frame

    def showVideo(self, videotype, sord, im):  # 封装展示结果的函数
        if (videotype == "front") & (sord == "src"):
            frame = self.trans(im, self.label_src)
            self.label_src.setPixmap(frame)
            self.label_src.setScaledContents(True)  # 设置图像自适应界面大小
        if (videotype == "front") & (sord == "det"):
            frame = self.trans(im, self.label_det)
            self.label_det.setPixmap(frame)
            self.label_det.setScaledContents(True)  # 设置图像自适应界面大小
        if (videotype == "back") & (sord == "src"):
            frame = self.trans(im, self.label_src2)
            self.label_src2.setPixmap(frame)
            self.label_src2.setScaledContents(True)  # 设置图像自适应界面大小
        if (videotype == "back") & (sord == "det"):
            frame = self.trans(im, self.label_det2)
            self.label_det2.setPixmap(frame)
            self.label_det2.setScaledContents(True)  # 设置图像自适应界面大小

    def clearThread(self, thread):  # 清除线程
        if thread.loadFrameThread is not None:
            thread.loadFrameThread.quit()
            thread.loadFrameThread.wait()
        thread.quit()
        thread.wait()
        thread.timer = None
        thread.mainwin = None
        thread.video = None
        thread.filename = None
        # list 与蓝色polygon重叠
        thread.list_overlapping_blue_polygon = None
        # list 与黄色polygon重叠
        thread.list_overlapping_yellow_polygon = None
        # 下行数量
        thread.down_count = None
        # 上行数量
        thread.up_count = None
        thread.isDone = None
        thread.output = None
        thread.polygon_mask_blue_and_yellow = None
        thread.color_polygons_image = None
        thread.objtracker = None
        thread.detector = None
        thread.framelist = None



    def closeApp(self):  # 退出应用程序
        if self.srcVideo is not None:
            self.srcVideo.release()
        if self.srcVideo_back is not None:
            self.srcVideo_back.release()
        if self.videoThread is not None:
            self.clearThread(self.videoThread)
        if self.videoThread_back is not None:
            self.clearThread(self.videoThread_back)
        app = QApplication.instance()
        app.quit()

    def startVideo(self):  # 开始按钮
        if self.sender() == self.start:  # 打开第一组（前门）
            if self.srcVideo is None:
                QMessageBox.critical(self, "错误", "还没有打开任何视频", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                return
            self.videoThread.timer.blockSignals(False)
            self.videoThread.timer.start(40)
        if self.sender() == self.start2:  # 打开第二组（后门）
            if self.srcVideo_back is None:
                QMessageBox.critical(self, "错误", "还没有打开任何视频", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                return
            self.videoThread_back.timer.blockSignals(False)
            self.videoThread_back.timer.start(40)

    def stopVideo(self):  # 暂停按钮
        if self.sender() == self.stop:  # 打开第一组（前门）
            if self.srcVideo is None:
                QMessageBox.critical(self, "错误", "还没有打开任何视频", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                return
            self.videoThread.timer.blockSignals(True)
        if self.sender() == self.stop2:  # 打开第二组（后门）
            if self.srcVideo_back is None:
                QMessageBox.critical(self, "错误", "还没有打开任何视频", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                return
            self.videoThread_back.timer.blockSignals(True)
    def clearAllNum(self):  # 清除数据
        self.lcdNumber_in.display(0)
        self.lcdNumber_out.display(0)
        self.lcdNumber_online.display(0)
        self.up_count = 0
        self.down_count = 0
        if self.videoThread is not None:
            self.videoThread.isDone = []
        if self.videoThread_back is not None:
            self.videoThread_back.isDone = []


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwin = Mainwin()
    mainwin.show()
    sys.exit(app.exec_())
