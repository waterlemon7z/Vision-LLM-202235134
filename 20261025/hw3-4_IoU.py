"""
 - Filename    :  hw3-4_IoU.py
 - Author      :  MinSeok Choi
 - Homework topic : IoU, NMS 구현
"""
import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
from random import randint, gauss

def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))
def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def calculate_iou(bbox_a, bbox_b):
    (xa_1, ya_1, xa_2, ya_2) = bbox_a
    (xb_1, yb_1, xb_2, yb_2) = bbox_b
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)

    intersection_area = w * h
    area_a = max(0, xa_2 - xa_1) * max(0, ya_2 - ya_1)
    area_b = max(0, xb_2 - xb_1) * max(0, yb_2 - yb_1)
    union_area = area_a + area_b - intersection_area

    return intersection_area / (union_area + 1e-6)

hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """

    pruned_boxes = []
    pruned_scores = []
    for (box, score) in zip(boxes, scores):
        if conf_threshold <= score:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """

    nms_boxes = []
    nms_scores = []
    sorted_scores_idx = np.argsort(scores)[::-1]
    while len(sorted_scores_idx) > 0:

        # 가장 높은 Confidence Score의 박스 선택
        idx = sorted_scores_idx[0]
        high_box = boxes[idx]

        # 선택된 박스는 최종 결과에 추가
        nms_boxes.append(high_box)
        nms_scores.append(scores[idx])

        # 남은 박스들의 인덱스
        remaining_idx = sorted_scores_idx[1:]

        # IoU가 Threshold 미만인 박스만 유지
        keep_idx = []

        for j in remaining_idx:
            cur_box = boxes[j]
            iou = calculate_iou(high_box, cur_box)

            if iou < iou_threshold:
                keep_idx.append(j)

        # 다음 반복에서 검사할 박스 갱신
        sorted_scores_idx = np.array(keep_idx, dtype=int)
    return nms_boxes, nms_scores

# print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(boxes, scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))

hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off")
plt.imsave('output/nms_output.png', hubble)