"""
Author  : MinSeok Choi
Date    : 2026-10-24
"""

import time
import cv2
from src.visionProcessing import HandVision
from src.gameCore import BasketGame
import sys

def main():
    # non-jetson 이 argv로 들어오면 일반적으로 카메라 불러오기
    if len(sys.argv) > 1 and sys.argv[1] == 'non-jetson':
        print("Not Jetson")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        pipeline = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "queue leaky=downstream max-size-buffers=2 "
            "max-size-bytes=0 max-size-time=0 ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # 손 탐지하는 클래스 인스턴스 생성
    hand_vision = HandVision()
    # 게임 코어 시스템 생성
    game = BasketGame(640,480)
    game.start()
    previous_time = time.perf_counter() # 치킨이 떨어지는 속도를 구현하기 위한 시간 변수

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.perf_counter()
            dt = now - previous_time #이전 프레임과 시간 차 비교
            previous_time = now

            # 웹캠으로 입력받은 frame을 전처리하고 손 인식
            frame, hand = hand_vision.process(frame)
            # 위에서 받은 손 데이터를 게임 코어 인스턴스에 업데이트
            game.update_cv_input(hand)
            # 게임 진행
            game.update(dt)

            # 게임과 관련된 asset 불러와서 화면에 띄우기
            frame = game.draw_display(frame)

            cv2.imshow("Basket Game", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("f"):
                # filter 모드 변경 부분
                hand_vision.filter_mode = (hand_vision.filter_mode + 1) % 5

            elif key == ord("q"):
                break

    finally:
        hand_vision.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
