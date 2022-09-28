import numpy as np
import objtracker
from objdetector import Detector
import cv2
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

VIDEO_PATH = './video/s+d.avi'

if __name__ == '__main__':



    # 打开视频
    capture = cv2.VideoCapture(VIDEO_PATH)
    originalSize = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 根据视频尺寸，填充供撞线计算使用的polygon
    width = 1920
    height = 1080
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)

    # 填充第一个撞线polygon（蓝色）
    list_pts_blue = [[0, 300], [0, 350], [1920, 350], [1920, 300]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个撞线polygon（黄色）
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    list_pts_yellow = [[0, 370], [0, 420], [1920, 420], [1920, 370]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    # polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (width//2, height//2))
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, originalSize)

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
    color_polygons_image = cv2.resize(color_polygons_image, originalSize)

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 下行数量
    down_count = 0
    # 上行数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int((width/2) * 0.01), int((height/2) * 0.05))

    # 实例化yolov5检测器
    detector = Detector()



    while capture.isOpened():
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break
        t1 = time.time()

        # 缩小尺寸，1920x1080->960x540
        # im = cv2.resize(im, (width//2, height//2))
        # im = cv2.resize(im, originalSize)


        list_bboxs = []
        # 更新跟踪器
        output_image_frame, list_bboxs = objtracker.update(detector, im)
        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)
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
                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    # 判断 黄polygon list里是否有此 track_id
                    # 有此track_id，则认为是下行方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 下行+1
                        down_count += 1
                        print('down count:', down_count, ', down id:', list_overlapping_yellow_polygon)
                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则认为是上行方向
                    if track_id in list_overlapping_blue_polygon:
                        # 上行+1
                        up_count += 1
                        print('up count:', up_count, ', up id:', list_overlapping_blue_polygon)
                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)
            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon   # 列表里存的都是碰到线到的id（无论上下车与否）
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)

                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
            list_overlapping_all.clear()
            # 清空list
            list_bboxs.clear()
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()

        # 输出计数信息
        text_draw = 'DOWN: ' + str(down_count) + \
                    ' , UP: ' + str(up_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=0.75, color=(0, 0, 255), thickness=2)
        t2 = time.time()
        print("时间：", t2-t1)
        cv2.imshow('Counting Demo', output_image_frame)
        cv2.waitKey(1)

    capture.release()
    cv2.destroyAllWindows()

