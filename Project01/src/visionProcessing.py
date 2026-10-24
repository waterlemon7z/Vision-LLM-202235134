
from pathlib import Path
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



class HandVision:
    """
    - name        : HandVision
    - description : 손 검출에 관련된 클래스
    :return:
    """
    def __init__(self):
        model_path = (
            Path(__file__).resolve().parent
            / "../models/MediaPipe/hand_landmarker.task"
        )
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            num_hands=1
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8)
        ] # 검지와 엄지만 시각화

        self.pinch_close = 0.35
        self.pinch_open = 0.50 # 집게 손가락이 열리고 닫힌 정도에 대한 수치

        self.pinching = False
        self.filtered_distance = None
        self.close_frames = 0
        self.open_frames = 0
        self.filter_mode = 0
        self.filter_names = ["ORIGINAL", "BLUR", "CLAHE", "CANNY", "SKIN"] #적용할 필터 종류

    def reset(self):
        self.pinching = False
        self.filtered_distance = None
        self.close_frames = 0
        self.open_frames = 0

    def process(self, frame):
        """
        - name        : process
        - description : 전처리 및 손 탐지 프로세스
        :return:
        """
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 손 화면 인식 부분 지정
        x1, y1 = int(w * 0), int(h * 0.5)
        x2, y2 = int(w * 1.), int(h * 1.)

        roi = frame[y1:y2, x1:x2]
        # CV 필터 시각화 모드 결정
        mode = self.filter_mode

        if mode == 1:  # Gaussian Blur
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (3, 3), 0)

        elif mode == 2:  # CLAHE
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            frame[y1:y2, x1:x2] = cv2.cvtColor(
                cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR
            )

        elif mode == 3:  # Canny Edge
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 70, 140)

            frame[y1:y2, x1:x2] = cv2.cvtColor(
                edges, cv2.COLOR_GRAY2BGR
            )

        elif mode == 4:  # Color Segmentation + Contour
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

            mask = cv2.inRange(
                ycrcb, (0, 133, 77), (255, 173, 127)
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (5, 5)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            segmented = cv2.bitwise_and(roi, roi, mask=mask)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) > 250:
                    cv2.drawContours(
                        segmented, [contour], -1, (0, 255, 0), 2
                    )

            frame[y1:y2, x1:x2] = segmented

        # 손 검출 시 정보를 저장할 dict
        info = {
            "detected": False,
            "pinching": False,
            "pinch_started": False,
            "center": None,
            "distance": None
        }

        color = (0, 0, 255)
        rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        # 손 검출
        result = self.detector.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
        )
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            points = [
                (
                    x1 + int(p.x * (x2 - x1)),
                    y1 + int(p.y * (y2 - y1))
                )
                for p in hand
            ]

            thumb_tip = points[4]
            index_tip = points[8]

            # 손 크기로 거리 정규화 (카메라 거리에 따른 길이 정규화)
            palm_size = max(math.dist(points[0], points[5]), 1)
            distance = math.dist(thumb_tip, index_tip) / palm_size

            # EMA 필터 (너무 값이 급변하면 떨려 보일 수 있기에 현재 값에 40%, 이전 값에 60% 곱해주자)
            if self.filtered_distance is None:
                self.filtered_distance = distance
            else:
                self.filtered_distance = (
                    0.6 * self.filtered_distance + 0.4 * distance
                )

            pinch_started = False

            # 집게 동작 판정
            if not self.pinching:
                self.close_frames = (
                    self.close_frames + 1
                    if self.filtered_distance < self.pinch_close
                    else 0
                )

                if self.close_frames >= 2:
                    self.pinching = True
                    pinch_started = True
                    self.close_frames = 0
            else:
                self.open_frames = (
                    self.open_frames + 1
                    if self.filtered_distance > self.pinch_open
                    else 0
                )

                if self.open_frames >= 2:
                    self.pinching = False
                    self.open_frames = 0

            center = (
                (thumb_tip[0] + index_tip[0]) // 2,
                (thumb_tip[1] + index_tip[1]) // 2
            )
            # 손 검출된 정보 저장
            info = {
                "detected": True,
                "pinching": self.pinching,
                "pinch_started": pinch_started,
                "center": center,
                "distance": self.filtered_distance,
                'finger_thumb': thumb_tip,
                'finger_index': index_tip,
            }

            color = (0, 255, 0) if self.pinching else (0, 255, 255)

            # 엄지 검지 시각화
            for start, end in self.connections:
                cv2.line(
                    frame, points[start], points[end],
                    (0, 255, 0), 2
                )

            for i in range(9):
                cv2.circle(
                    frame, points[i],
                    6 if i in (4, 8) else 4,
                    (0, 0, 255) if i in (4, 8) else (255, 0, 0),
                    -1
                )

            cv2.line(frame, thumb_tip, index_tip, color, 2)
            cv2.circle(frame, center, 7, color, -1)

        else:
            self.reset()

        # 현재 손 인식 상태 표시
        status = (
            "NO HAND" if not info["detected"]
            else "PINCH" if info["pinching"]
            else "OPEN"
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 현재 손 모양 출력
        cv2.putText(
            frame, status, (400, 440),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
        # 현재 필터 정보에 대한 출력
        cv2.putText(
            frame,
            f"Filter: {self.filter_names[self.filter_mode]}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        return frame, info

    def close(self):
        self.detector.close()
