"""Defines related function to process defined data structure."""
import imp
import math
import numpy as np
import torch
import math
import cv2
from collections import namedtuple
from enum import Enum
import config

MarkingPoint = namedtuple('MarkingPoint', ['x',
                                           'y',
                                           'lenSepLine_x',
                                           'lenSepLine_y',
                                           'lenEntryLine_x',
                                           'lenEntryLine_y'])

def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2 * math.pi - diff

def calc_point_direction_angle(point_a, point_b):
    """Calculate angle between direction in rad."""
    return direction_diff(point_a.direction, point_b.direction)

def calc_line_direction_angle(line_gt, line_pd, line_type='entry'):
    if line_type == 'entry':
        direc_a = math.atan2(line_gt.lenEntryLine_y, line_gt.lenEntryLine_x)
        direc_b = math.atan2(line_pd.lenEntryLine_y, line_pd.lenEntryLine_x)
    elif line_type == 'sep':
        direc_a = math.atan2(line_gt.lenSepLine_y, line_gt.lenSepLine_x)
        direc_b = math.atan2(line_pd.lenSepLine_y, line_pd.lenSepLine_x)
    else:
        raise ValueError(f'Wrong eval type: {line_type}')
    return direction_diff(direc_a, direc_b)

def calc_line_dist(gt, pd, axis_type='x'):
    ''' ground truth & predicted 小于半个车位短边'''
    if axis_type == 'x':
        delta_dist = abs(pd.lenEntryLine_x - gt.lenEntryLine_x)
    elif axis_type == 'y':
        delta_dist = abs(pd.lenEntryLine_y - gt.lenEntryLine_y)
    else:
        raise ValueError(f'Wrong eval type: {axis_type}')
    return delta_dist


def calc_point_squre_dist_v1(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    # res = (distx/3)**2 + (disty*1)**2
    # res = distx**2 + (disty*1)**2
    res = distx**2 + (disty)**2 # H:W = 3:1
    return res


def calc_point_squre_dist_v2(gt, pd):
    gt_x = gt.x + gt.lenEntryLine_x
    gt_y = gt.y + gt.lenEntryLine_y
    pd_x = pd.x + pd.lenEntryLine_x
    pd_y = pd.y + pd.lenEntryLine_y
    # y * 3, 相对 x 来说，以短边为基准；更严格
    return (gt_x - pd_x)**2 + (gt_y - pd_y)**2


def match_marking_points(ground, predict):
    """Determine whether a detected point match ground truth.
        # params.squared_distance_thresh = 0.005
        # params.direction_angle_thresh = 0.5235987755982988  
    """
    entry_startpoint = calc_point_squre_dist_v1(ground, predict)
    entry_endpoint   = calc_point_squre_dist_v2(ground, predict)
    angle_entry = calc_line_direction_angle(ground, predict, 'entry')
    angle_sep   = calc_line_direction_angle(ground, predict, 'sep')
    # 3 for 30 degrees and 1 for 10 degrees
    ratio = 1
    # ratio = 2
    # ratio = 3
    # ratio = 6
    # mode 1 strict
    effect_flag = entry_startpoint < config.SQUARED_DISTANCE_THRESH \
                  and entry_endpoint < config.SQUARED_DISTANCE_THRESH \
                  and angle_sep < config.DIRECTION_ANGLE_THRESH / ratio \
                  and angle_entry < config.DIRECTION_ANGLE_THRESH / ratio
    # mode 2 loose
    # effect_flag = entry_startpoint < params.squared_distance_thresh # 0.005
    # effect_flag = entry_endpoint < params.squared_distance_thresh
    # effect_flag = entry_startpoint < params.squared_distance_thresh \
    #               and angle_sep < params.direction_angle_thresh / ratio
    # effect_flag = entry_startpoint < params.squared_distance_thresh \
    #               and entry_endpoint < params.squared_distance_thresh
    return effect_flag


def non_maximum_suppression(pred_points):
    """Perform non-maxmum suppression on marking points."""
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            i_x = pred_points[i][1].x
            i_y = pred_points[i][1].y
            j_x = pred_points[j][1].x
            j_y = pred_points[j][1].y
            if abs(j_x - i_x) < 1 / config.FEATURE_MAP_SIZE and abs(
                    j_y - i_y) < 1 / config.FEATURE_MAP_SIZE:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points


def get_predicted_points(predictions,thresh):
    """Get marking points from one predicted feature map."""
    # 传进来的batch=1
    assert isinstance(predictions, torch.Tensor)
    predicted_points = []
    predictions = predictions.detach().cpu().numpy()
    batchsize,C, feature_H, feature_W = predictions.shape
    assert C == 7
    # thresh = config.CONFID_THRESH_FOR_POINT # 0.01
    result = []

    for batch in range(batchsize):
        prediction = predictions[batch,...]
        predicted_points = []

        for i in range(feature_H):
            for j in range(feature_W):
                if prediction[0, i, j] >= thresh:
                    obj_x = (j + prediction[1, i, j]) / feature_W
                    obj_y = (i + prediction[2, i, j]) / feature_H
                    lenEntryLine_x = prediction[3, i, j]
                    lenEntryLine_y = prediction[4, i, j]
                    lenSepLine_x = prediction[5, i, j]
                    lenSepLine_y = prediction[6, i, j]
                    # isOccupied = prediction[7, i, j]
                    marking_point = MarkingPoint(obj_x, obj_y,
                                                 lenSepLine_x, lenSepLine_y,
                                                 lenEntryLine_x, lenEntryLine_y)
                    predicted_points.append((prediction[0, i, j], marking_point))
        result.append(non_maximum_suppression(predicted_points))

    # return non_maximum_suppression(predicted_points, params)
    return result



def plot_slots(image, eval_results, params, img_name=None):
    """
    画进入线目标点：逆时针旋转车位的进入线起始端点。
    AB-BC-CD-DA 这里指A点
       Parking Slot Example
            A.____________D
             |           |
           ==>           |
            B____________|C

     Entry_line: AB
     Separable_line: AD (regressive)
     Separable_line: BC (un-regressive, calc)
     Object_point: A (point_0)
      cos sin theta 依据的笛卡尔坐标四象限
               -y
                |
             3  |  4
        -x -----|-----> +x (w)
             2  |  1
                ↓
               +y (h)
    """
    pred_points =  eval_results['pred_points'] \
        if 'pred_points' in eval_results else eval_results
    if not pred_points:
        return image
    height = image.shape[0]
    width = image.shape[1]
    for confidence, marking_point in pred_points:
        if confidence < params.confid_plot_inference:
            continue # DMPR-PS是0.5
        x, y, lenSepLine_x, lenSepLine_y, \
        lenEntryLine_x, lenEntryLine_y = marking_point[:6]
        # p0->p1为进入线entry_line
        # p0->p3为分隔线separable_line
        # p1->p3也为分割线
        # 上述箭头"->"代表向量方向，p0->p1即p0为起点，p3为终点，p0指向p3
        p0_x = width * x - 1
        p0_y = height * y - 1
        p1_x = p0_x + width * lenEntryLine_x
        p1_y = p0_y + height * lenEntryLine_y
        length = 300
        # sepline direction version
        H, W = params.input_size
        x_ratio, y_ratio = 1, H / W
        radian = math.atan2(marking_point.lenSepLine_y * y_ratio,
                            marking_point.lenSepLine_x * x_ratio)
        sep_cos = math.cos(radian)
        sep_sin = math.sin(radian)
        p3_x = int(p0_x + length * sep_cos)
        p3_y = int(p0_y + length * sep_sin)
        p2_x = int(p1_x + length * sep_cos)
        p2_y = int(p1_y + length * sep_sin)
        p0_x, p0_y = round(p0_x), round(p0_y)
        p1_x, p1_y = round(p1_x), round(p1_y)
        # 画进入线目标点：逆时针旋转车位的进入线起始端点。
        # AB-BC-CD-DA 这里指A点
        cv2.circle(image, (p0_x, p0_y), 5, (0, 0, 255), thickness=2)
        # 给目标点打上置信度，取值范围：0到1
        color = (255, 255, 255) if confidence > 0.7 else (100, 100, 255)
        if confidence < 0.3: color = (0, 0, 255)
        cv2.putText(image, f'{confidence:.3f}', # 不要四舍五入
                    (p0_x + 6, p0_y - 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, color)
        # 画上目标点坐标 (x, y)
        cv2.putText(image, f' ({p0_x},{p0_y})',
                    (p0_x, p0_y + 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # 在图像左上角给出图像的分辨率大小 (W, H)
        H, W = params.input_size
        cv2.putText(image, f'({W},{H})', (5, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # 画车位进入线 AB
        cv2.arrowedLine(image, (p0_x, p0_y), (p1_x, p1_y), (0, 255, 0), 2, 8, 0, 0.2)
        # 画车位分割线 AD
        cv2.arrowedLine(image, (p0_x, p0_y), (p3_x, p3_y), (255, 0, 0), 2) # cv2.line
        # 画车位分割线 BC
        if p1_x >= 0 and p1_x <= width - 1 and p1_y >= 0 and p1_y <= height - 1:
            cv2.arrowedLine(image, (p1_x, p1_y), (p2_x, p2_y), (33, 164, 255), 2)

    return image