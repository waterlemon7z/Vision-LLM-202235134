"""
Author  : MinSeok Choi
Date    : 2026-11-07
File     : src/audio_system.py
"""
import platform
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
OS = platform.system()
if OS == "Darwin":
    import sounddevice as sd
    import soundfile as sf

MIC_DEVICE = "plughw:3,0"
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "../whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "../whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
PIPER_PYTHON = "../.piper_venv/bin/python"
PIPER_MODEL = "../src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"
ALARM_FILE = "src/audio/alarm.wav"

class AudioSystem:
    """
    - name        : AudioSystem
    - description : 오디오와 관련된 작업을 수행하는 클래스
    """
    _STT_MARKERS = re.compile(
        r"\[\s*(?:끝|무음|침묵|잡음|음악|박수|음성\s*없음|"
        r"BLANK_AUDIO|NO_SPEECH|MUSIC|APPLAUSE)\s*\]",
        re.IGNORECASE,
    )

    def __init__(self):
        if not Path(ALARM_FILE).is_file():
            raise FileNotFoundError(f"경보 파일이 없습니다: {ALARM_FILE}")
        # 알람 재생을 비동기로 하기 위한 스레드 관련 변수
        self._alarm_stop = Event() # 스레드 간에 신호를 주고받을 때 사용함
        self._alarm_thread = None
        self._recording = Event()
        self._tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    def is_alarm_active(self):
        """
        - name        : is_alarm_active
        - description : 알람이 현재 실행 중인가?
        :return bool:
        """
        return self._alarm_thread is not None and self._alarm_thread.is_alive()

    def start_alarm(self):
        """
        - name        : start_alarm
        - description : 알람 시작 함수
        :return None:
        """
        if self.is_alarm_active():
            return
        self._alarm_stop.clear()
        self._alarm_thread = Thread(target=self._play_alarm_loop, name="alarm-player")
        self._alarm_thread.start()

    def stop_alarm(self):
        """
        - name        : stop_alarm
        - description : 알람 멈추기 함수
        :return None:
        """
        self._alarm_stop.set()
        if self._alarm_thread is not None:
            self._alarm_thread.join()
            self._alarm_thread = None

    def _play_alarm_loop(self):
        """
        - name        : _play_alarm_loop
        - description : 경보기 알람을 재생하는 루프. F키를 눌렀는지 수시로 확인하여 경보기 알람 유지를 결정한다.
        :return None:
        """
        player = "afplay" if OS == "Darwin" else "aplay"
        process = None
        try:
            while not self._alarm_stop.is_set():
                process = subprocess.Popen([player, str(ALARM_FILE)])
                # 재생 도중에도 f키에 의한 중지 요청을 확인한다.
                while process.poll() is None:
                    if self._alarm_stop.wait(0.05):
                        return
                if process.returncode != 0:
                    raise RuntimeError(f"경보 재생 실패: 종료 코드 {process.returncode}")
        except (OSError, RuntimeError) as exc:
            print(f"Alarm error: {exc}")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


    def __record_audio(self):
        """
        - name        : __record_audio
        - description : 녹음 시작하는 private 함수
        :return None:
        """
        print(f"녹음 시작 ({OS})")

        if OS == "Darwin":
            # macOS: 기본 마이크 사용
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            sf.write(str(AUDIO_FILE), audio, SAMPLE_RATE)

        elif OS == "Linux":
            # Linux / Jetson: ALSA 마이크 사용
            subprocess.run(
                [
                    "arecord",
                    "-D", MIC_DEVICE,
                    "-f", "S16_LE",
                    "-r", str(SAMPLE_RATE),
                    "-c", "1",
                    "-d", str(RECORD_SECONDS),
                    str(AUDIO_FILE),
                ],
                check=True,
            )
        print("녹음 완료")
        return AUDIO_FILE


    def speech_to_text(self):
        """
        - name        : speech_to_text
        - description : stt, 예제코드와 동일
        :return None:
        """
        self._recording.set()
        try:
            audio_path = self.__record_audio()
        finally:
            self._recording.clear()

        result = subprocess.run(
            [
                str(WHISPER_PATH),
                "-m", str(WHISPER_MODEL),
                "-f", str(audio_path),
                "-l", "ko",
                "--no-timestamps",
            ],
            text=True,
            capture_output=True,
            check=True,
        )

        raw_text = result.stdout.strip()
        text = self.clean_stt_text(raw_text)
        if raw_text and not text:
            print(f"STT 무효 결과 무시: {raw_text}")
        return text

    def is_recording(self):
        return self._recording.is_set()

    @classmethod
    def clean_stt_text(cls, text):
        """
        - name        : clean_stt_text
        - description : stt로 나온 문자열 정제
        :return None:
        """
        cleaned = " ".join(cls._STT_MARKERS.sub(" ", text).split())
        return cleaned if any(char.isalnum() for char in cleaned) else ""


    def text_to_speech(self, text):
        """
        - name        : text_to_speech
        - description : tts 재생하면 main 동기 걸리니깐 스레드에 올리고 반환
        :return None:
        """
        future = self._tts_executor.submit(self._text_to_speech, text)
        return future

    def _text_to_speech(self, text):
        """
        - name        : _text_to_speech
        - description : tts 본체, 예제 코드
        :return None:
        """
        subprocess.run(
            [
                str(PIPER_PYTHON),
                "-m", "piper",
                "-m", str(PIPER_MODEL),
                "-f", str(OUTPUT_FILE),
            ],
            input=text,
            text=True,
            check=True,
        )
        # 운영체제별
        if OS == "Darwin":
            subprocess.run(
                ["afplay", str(OUTPUT_FILE)],
                check=True,
            )

        elif OS == "Linux":
            subprocess.run(
                ["aplay", str(OUTPUT_FILE)],
                check=True,
            )

    def close(self):
        """종료임"""
        self.stop_alarm()
        self._tts_executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__": # 예시 실행 부분
    audio_system = AudioSystem()
    try:
        question = audio_system.speech_to_text()
        print(question)

        if question:
            audio_system.text_to_speech(question).result()
    finally:
        audio_system.close()
