"""
Author  : MinSeok Choi
Date    : 2026-11-07
File     : main.py
"""
from concurrent.futures import ThreadPoolExecutor # 스레드를 위한 라이브러리
import cv2
import sys
# 사용자 정의 py
from src.local_llm import Gemma4
from src.audio_system import AudioSystem, RECORD_SECONDS
from src.local_yolo import ObjectTracker


def main():
    """
    - name        : main
    - description : 카메라를 받아오고 while loop을 돌면서 yolo(동기), whisper, piper, gemma4(비동기)들을 실행합니다.
                    s, f키에 각각 역할이 할당되어있으며 도난 방지 시스템의 기본 flow를 구성하여 실행합니다.
    :return none:
    """
    # non-jetson 이 argv로 들어오면 일반적으로 카메라 불러오기
    if len(sys.argv) > 1 and sys.argv[1] == 'non-jetson':
        print("Not Jetson")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        pipeline = (
            "v4l2src device=/dev/video0 ! "
            "image/jpeg,width=1280,height=720,framerate=30/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "queue leaky=downstream max-size-buffers=2 "
            "max-size-bytes=0 max-size-time=0 ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # Phase1-1: 각 클래스 호출하여 인스턴스 생성
    obj_tracker = ObjectTracker() # Yolo를 통한 object 트래킹
    audio_system = AudioSystem() # Audio 시스템 전반 컨트롤 (tts / stt)
    gemma_llm = Gemma4() # gemma4 사용을 위한 클래스
    # Phase1-2: 스레드에 필요한 변수 선언
    thread_executor = ThreadPoolExecutor(max_workers=1) # 스레드 실행기
    speech_future = None # 스레드 상태 저장 변수
    gemma_future = None # 스레드 상태 저장 변수
    protection_request_cancelled = False # 스레드 처리 중에 프로그램 종료 요청

    # Phase2: 본격 프로그램 시작
    try: # 프로그램 종료 혹은 충돌 시 finally로~
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            obj_tracker.run(frame, render=False)
            items = obj_tracker.get_items()

            # Phase3-2: s키가 눌러진 뒤 speech_future 스레드가 종료되어 stt가 완료된 때.
            if speech_future is not None and speech_future.done():
                try:
                    protection_text = speech_future.result() # 스레드에서 return 된 값 가져오기
                    print(f"Protect call text: {protection_text}" if protection_text else "No record")
                    if protection_text and not protection_request_cancelled: # 만약에 중간에 종료가 안되었고, text가 반환되었으면
                        detected_classes = list(dict.fromkeys(items))
                        print(f"Gemma input classes: {detected_classes}")
                        # Phase4-1: Gemma에게 넘긴다.
                        gemma_future = thread_executor.submit(gemma_llm.generate, protection_text, detected_classes)
                except Exception as exc:
                    print(f"Recognize record error: {exc}")
                finally:
                    speech_future = None

            # Phase 4-2: Gemma의 추론이 끝난 때.
            if gemma_future is not None and gemma_future.done():
                try:
                    protect_item_dict = gemma_future.result() # gemma의 추론 결과 저장
                    print(protect_item_dict)
                    if not protection_request_cancelled: # 만약에 도중에 f 키가 눌리지 않았다면
                        if protect_item_dict:
                            locked_objects = obj_tracker.lock_on(protect_item_dict) # 감시 대상인 아이템 반환
                            print(f"Locked object IDs: {list(locked_objects)}")
                            speech = ", ".join(obj["korean_name"] for obj in locked_objects.values()) + "에 대한 보호가 시작되었습니다."
                            print(f"Play Audio: {speech}")
                            audio_system.text_to_speech(speech) # 오디오 플레이
                        else:
                            print("I cannot find items to protect")
                except Exception as exc:
                    print(f"gemma4 result error: {exc}")
                finally:
                    gemma_future = None

            # Phase 5: 보호를 부탁한 아이템이 잘 계신지 확인 (3초)
            protection_states = obj_tracker.check_protected_objects() #각 보호 대상인 object의 state 반환
            missing_ids = {track_id for track_id, state in protection_states.items() if state["missing"]} # missing인 object 반환
            # Phase 5-a:만약 분실된 아이템이 있으면?
            if missing_ids:
                for track_id in missing_ids:
                    state = protection_states[track_id]
                    print(f"Protected item vanished: {state['korean_name']} (ID: {track_id})")
                missing_names = ", ".join(protection_states[track_id]["korean_name"] for track_id in sorted(missing_ids))
                print(f"Missing names: {missing_names}")
                obj_tracker.update_protect_status(False)
                protection_request_cancelled = True
                audio_system.text_to_speech("도둑 맞음.")
                print('물건 털림')
                audio_system.start_alarm()

            # Phase 5-b: 아이템이 잘 있으면 그냥 pass

            # 현재 상태에 대한 text 설정 구역
            activity_text = None
            if speech_future is not None and not protection_request_cancelled:
                activity_text = (f"녹음 시작 ({RECORD_SECONDS}초)" if audio_system.is_recording()
                                 else "음성 인식 중")
            output_frame = obj_tracker.render_frame(frame, activity_text=activity_text,
                                                    alarm_active=audio_system.is_alarm_active())

            cv2.imshow("Item protector", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Phase3-1: 녹음 및 STT 시작 트리거
                if audio_system.is_alarm_active(): # 소잃고 외양간 고치는 경우
                    print("Stop alarm to press key F first")
                elif obj_tracker.is_protecting(): # 이미 보호 중인 경우
                    print("Already protected")
                elif speech_future is None and gemma_future is None:
                    protection_request_cancelled = False
                    speech_future = thread_executor.submit(audio_system.speech_to_text)
                else:
                    print("Record processing..")
            elif key == ord("f") and (obj_tracker.is_protecting() or audio_system.is_alarm_active()):
                # 보안 해제 구역 (원상태로 복구)
                audio_system.stop_alarm()
                obj_tracker.update_protect_status(False)
                # 진행 중인 STT/Gemma 결과와 무관하게 종료하기 위한 변수
                protection_request_cancelled = True
                audio_system.text_to_speech("물건 보호를 중지했습니다.")
                print('Protecting system down')

    finally:
        audio_system.stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        # 스레드 정리
        thread_executor.shutdown(wait=True, cancel_futures=True)
        audio_system.close()
if __name__ == "__main__":
    main()
