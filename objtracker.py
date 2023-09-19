# 封装了一个目标追踪器，对检测的物体进行追踪
import json
import threading

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy as np
import orjson
from feature_extractor_net.feature_extractor import Extractor
from net_part.mqtt_publish import *
class Objtracker():
    def __init__(self):
        cfg = get_config()
        cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        # 存放每个id第一次出现的特征序列
        self.features = {}
        self.ex = Extractor("./feature_extractor_net/checkpoint/ckpt.t7")
        self.client = connect_mqtt()
        self.client.loop_start()
        self.timer = threading.Timer(1, self.send_timer_callback)
        self.timer.daemon = True  # 将定时器设为守护线程，这样它会随着主线程的退出而退出
        self.timer.start()

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        list_pts = []
        point_radius = 4

        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            if cls_id in ['smoke', 'phone', 'eat']:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            if cls_id == 'eat':
                cls_id = 'eat-drink'

            # check whether hit line
            check_point_x = x1
            check_point_y = int(y1 + ((y2 - y1) * 0.6))

            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
            list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
            list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
            list_pts.append([check_point_x+point_radius, check_point_y-point_radius])

            ndarray_pts = np.array(list_pts, np.int32)
            cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))  # 小红点
            list_pts.clear()
        return image

    def update(self, target_detector, image):
            _, bboxes = target_detector.detect(image)
            bbox_xywh = []

            # lbl_conf = []
            confs = []
            bboxes2draw = []
            if len(bboxes):
                # Adapt detections to deep sort input format
                for x1, y1, x2, y2, lbl, conf in bboxes:
                    obj = [
                        int((x1+x2)/2), int((y1+y2)/2),
                        x2-x1, y2-y1
                    ]
                    bbox_xywh.append(obj)
                    # lbl_conf.append((lbl, conf))
                    confs.append(conf)
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = self.deepsort.update(xywhs, confss, image)
                for value in list(outputs):
                    x1,y1,x2,y2,track_id = value
                    bboxes2draw.append(
                        (x1, y1, x2, y2, "",track_id)
                    )
                    if f"{track_id}" not in self.features:
                        crop_feature = self.getFeatures(value, image)
                        # data = {f"{track_id}": crop_feature}
                        # send_data = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
                        # # 发布模板数据
                        # publish_msg(self.client, send_data)
            image = self.plot_bboxes(image, bboxes2draw)
            return image, bboxes2draw

    # 获取扣图部分的特征
    def getFeatures(self, bboxInfo, im):
        x1, y1, x2, y2, track_id = bboxInfo
        im_crop = im[np.newaxis, y1:y2, x1:x2]  # 抠图部分
        crop_feature = self.ex(im_crop)
        self.features[f"{track_id}"] = crop_feature[0]
        cv2.imwrite(f"./result/{track_id}.jpg", im[y1:y2, x1:x2])
        return crop_feature[0]

    def send_timer_callback(self):
        print("--------------------------")
        self.timer = threading.Timer(1, self.send_timer_callback)
        self.timer.daemon = True  # 将定时器设为守护线程，这样它会随着主线程的退出而退出
        self.timer.start()
        # 发布模板数据
        if self.features:
            send_data = orjson.dumps(self.features, option=orjson.OPT_SERIALIZE_NUMPY)
            publish_msg(self.client, send_data)


