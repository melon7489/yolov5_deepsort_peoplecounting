# 封装的一个目标检测器，对视频中的物体进行检测
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device
import objtracker

OBJ_LIST = ['car','person']
# OBJ_LIST = ['head']
DETECTOR_PATH = 'weights/yolov5s.pt'
objtracker = objtracker.Objtracker()
class baseDet(object):
    def __init__(self):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):
        self.frameCounter = 0

    def feedCap(self, im, func_status):
        retDict = {
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }
        self.frameCounter += 1
        im, obj_bboxes = objtracker.update(self, im)
        # objtracker.send_timer()
        retDict['frame'] = im
        retDict['obj_bboxes'] = obj_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")


class Detector(baseDet):
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, self.device)
        model.to(self.device).eval()
        # model.half()
        model.float()
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        # img = img.half()  # 半精度
        img = img.float()
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detect(self, im):
        im0, img = self.preprocess(im)
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in OBJ_LIST:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
        return im, pred_boxes

