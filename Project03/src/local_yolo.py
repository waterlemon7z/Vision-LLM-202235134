"""
Author  : MinSeok Choi
Date    : 2026-11-07
File     : src/local_yolo.py
"""
import time
import cv2
import numpy as np
from ultralytics import YOLO
from src.guard_display import GuardDisplay

YOLO_PATH = "../src/models/YOLO/yolo11n.pt"
class ObjectTracker:
    """
    - name        : ObjectTracker
    - description : 물체를 추적, Yolo 사용 클래스
    :return None:
    """
    def __init__(self, confirm_seconds=3.0, disappear_seconds=3.0,
                 candidate_gap_seconds=0.5):

        self.model = YOLO(YOLO_PATH)
        try:
            self.model.to("cuda")
        except: # 맥북에서 실행 (간단 코드)
            self.model.to("mps")

        self.object_list_ns = [] #n초 동안 유지된 물체 저장 리스트
        self.confirm_seconds = confirm_seconds # 물체로 인정할 시간 초
        self.disappear_seconds = disappear_seconds # 사라진 물체로 인정할 시간 초
        self.candidate_gap_seconds = candidate_gap_seconds # 재탐지로 유지할 최대 시간
        self._objects = {}  # track_id별 후보 및 등록 객체, 보호 대상은 별도 관리한다
        self.__protecting = False
        self.protected_objects = {}  # 감지 목록에서 사라져도 유지하는 track_id별 보호 기록
        self._current_track_ids = set()
        self._display = GuardDisplay() # frame 옆에 정보를 띄워주는 클래스
        self._detections = []

    def run(self, frame, render=True):
        results = self.model.track(
            source=frame,  # source image
            persist=True,  # 연속 프레임에서 추적 ID 유지
            tracker="bytetrack.yaml",
            conf=0.25,  # Confidence Threshold
            iou=0.5,  # IoU Threshold
            verbose=False,  # no output prints
            classes=None,  # selected class
        )

        # 필요한 정보만 가져와서 frame에 출력
        result = results[0]
        detections = []
        boxes = result.boxes
        if boxes is not None:
            class_ids = boxes.cls.int().cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(class_ids)
            bboxes = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            for track_id, class_id, bbox, confidence in zip(
                    track_ids, class_ids, bboxes, confidences):
                detections.append({
                    "track_id": track_id,
                    "class_name": result.names[class_id],
                    "bbox": tuple(bbox),
                    "confidence": confidence,
                })

        # 탐지가 없는 프레임에서도 만료 처리를 해야 한다.
        self._detections = detections
        self._update_objects([obj for obj in detections if obj["track_id"] is not None], time.monotonic())
        return self.render_frame(frame) if render else frame

    def render_frame(self, frame, activity_text=None, alarm_active=False):
        return self._display.render(frame, self._detections,
                                    self.protected_objects, self.is_protecting(), activity_text,
                                    alarm_active=alarm_active)

    def get_items(self)->list:
        """
        - name        : get_items
        - description : 현재 감지되고 잇는 아이템 반환
        :return None:
        """
        return np.unique([obj["class_name"] for obj in self.object_list_ns]).tolist()

    def _update_objects(self, detections, now):
        """
        - name        : _update_objects
        - description : 3초 확인 후 등록하고, 마지막 탐지로부터 3초가 지나면 제거한다. 각 오브젝트 최신 상태 업데이트
        :return None:
        """
        self._current_track_ids = {obj["track_id"] for obj in detections}
        for detection in detections:
            protected = self.protected_objects.get(detection["track_id"])
            if protected is not None:
                protected.update(detection)
                protected["last_seen"] = now

        for track_id, obj in list(self._objects.items()):
            elapsed = now - obj["last_seen"]
            if (obj["confirmed"] and elapsed >= self.disappear_seconds) or (
                    not obj["confirmed"] and elapsed > self.candidate_gap_seconds):
                del self._objects[track_id]

        for detection in detections:
            track_id = detection["track_id"]
            if track_id not in self._objects:
                self._objects[track_id] = {
                    "first_seen": now,
                    "confirmed": False,
                }

            obj = self._objects[track_id]
            obj.update(detection)
            obj["last_seen"] = now
            if now - obj["first_seen"] >= self.confirm_seconds:
                obj["confirmed"] = True

        # 리스트 자체는 유지하고, 내부 상태와 분리된 최신 스냅샷을 제공한다.
        self.object_list_ns[:] = [
            obj.copy() for obj in self._objects.values() if obj["confirmed"]
        ]

    def lock_on(self, protect_item_dict: dict[str, str]):
        """
        - name        : lock_on
        - description : gemma4로 읽어온 아이템과 class를 일치시켜 물체가 화면에 존재하는지 검사를 위한 lock
        :return None:
        """

        targets = {
            obj["track_id"]: {**obj, "korean_name": protect_item_dict[obj["class_name"]]}
            for obj in self.object_list_ns
            if obj["class_name"] in protect_item_dict
            and obj["track_id"] in self._current_track_ids
        }
        unavailable = set(protect_item_dict) - {obj["class_name"] for obj in targets.values()}
        if unavailable:
            raise ValueError(f"현재 확인할 수 없는 보호 클래스: {', '.join(sorted(unavailable))}")

        self.protected_objects = targets
        self.update_protect_status(True)
        return {track_id: obj.copy() for track_id, obj in targets.items()}

    def check_protected_objects(self, now=None):
        """
        - name        : check_protected_objects
        - description : id 별로 사라진 시간을 검사
        :return None:
        """
        if now is None:
            now = time.monotonic()
        states = {}
        for track_id, obj in self.protected_objects.items():
            present = track_id in self._current_track_ids
            missing_seconds = 0.0 if present else max(0.0, now - obj["last_seen"])
            states[track_id] = {
                **obj,
                "present": present,
                "missing_seconds": missing_seconds,
                "missing": not present and missing_seconds >= self.disappear_seconds,
            }
        return states

    def is_protecting(self):
        return self.__protecting

    def update_protect_status(self, status:bool):
        """
        - name        : update_protect_status
        - description : 현재 보호 상태 업데이트
        :return bool:
        """
        if status and not self.protected_objects:
            raise ValueError("먼저 lock_on()으로 보호 대상을 지정해야 합니다.")
        self.__protecting = status
        if not status:
            self.protected_objects.clear()
        return self.__protecting

if __name__ == '__main__': # 예시 실행 부분
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    obj_tracker = ObjectTracker()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_frame = obj_tracker.run(frame)
            cv2.imshow("yolo test", output_frame)
            key = cv2.waitKey(1) & 0xFF
            print(obj_tracker.object_list_ns)
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
