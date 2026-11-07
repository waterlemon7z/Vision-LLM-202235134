"""
Author  : MinSeok Choi (codex)
Date    : 2026-11-07
File     : src/guard_display.py
"""
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class GuardDisplay:
    """
    - name        : GuardDisplay
    - description : frame옆에 현재 상태에 대한 표기, 구조 제외 codex 작성.
    """
    def __init__(self, panel_width=380):
        self.panel_width = panel_width
        font_path = Path(__file__).resolve().parent / "fonts/NanumGothic-Regular.ttf"
        self.label_font = ImageFont.truetype(str(font_path), 19)
        self.status_font = ImageFont.truetype(str(font_path), 24)

    @staticmethod
    def _wrap(text, font, width):
        lines = []
        line = ""
        for char in text:
            if line and font.getlength(line + char) > width:
                lines.append(line)
                line = char
            else:
                line += char
        if line:
            lines.append(line)
        return lines

    def render(self, frame, detections, protected_objects, protecting, activity_text=None, alarm_active=False):
        height, width = frame.shape[:2]
        counts = Counter(
            (obj["class_name"], obj["korean_name"])
            for obj in protected_objects.values()
        ) if protecting else Counter()
        names = ", ".join(
            f"{korean}" + (f" ×{count}" if count > 1 else "")
            for (_, korean), count in counts.items()
        )
        if alarm_active:
            status = "Status: 경보 발생"
            status_color = (248, 71, 71)
        elif protecting:
            status = f"Status: 보호 중 - {names}"
            status_color = (74, 222, 128)
        else:
            status = "Status: 보호 중지"
            status_color = (203, 213, 225)
        text_width = self.panel_width - 40
        status_lines = self._wrap(status, self.status_font, text_width)
        activity_lines = self._wrap(activity_text, self.label_font, text_width) if activity_text else []
        item_lines = []
        for (english, korean), count in counts.items():
            label = f"{english} / {korean}" + (f" ×{count}" if count > 1 else "")
            item_lines.extend(self._wrap(label, self.label_font, text_width))

        # 대상이 많으면 아래로 확장하여 상태와 목록이 잘리지 않게 한다.
        canvas_height = max(height, 160 + len(status_lines) * 34
                            + (len(item_lines) + len(activity_lines)) * 28)
        image = Image.new("RGB", (width + self.panel_width, canvas_height), (18, 23, 32))
        image.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 0))
        draw = ImageDraw.Draw(image)

        for obj in detections:
            if protecting:
                protected = protected_objects.get(obj["track_id"])
                if protected is None:
                    continue
                label = f"{obj['class_name']} / {protected['korean_name']}"
                color = (74, 222, 128)
            else:
                label = obj["class_name"]
                color = (96, 165, 250)

            x1, y1, x2, y2 = obj["bbox"]
            x1, x2 = sorted((max(0, min(width - 1, int(x1))), max(0, min(width - 1, int(x2)))))
            y1, y2 = sorted((max(0, min(height - 1, int(y1))), max(0, min(height - 1, int(y2)))))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            # 긴 클래스명도 영상 영역 안에서 줄바꿈한다.
            lines = self._wrap(label, self.label_font, max(1, width - 16))
            label_width = min(width, int(max(self.label_font.getlength(line) for line in lines)) + 12)
            label_height = len(lines) * 25 + 6
            left = min(x1, max(0, width - label_width))
            top = max(0, y1 - label_height)
            draw.rectangle((left, top, left + label_width - 1, min(height - 1, top + label_height)), fill=(18, 23, 32))
            for index, line in enumerate(lines):
                if top + 3 + index * 25 + 25 <= height:
                    draw.text((left + 6, top + 3 + index * 25), line,
                              font=self.label_font, fill=color)

        panel_left = width + 20
        draw.line((width, 0, width, canvas_height), fill=(51, 65, 85), width=2)
        y = 24
        for line in status_lines:
            draw.text((panel_left, y), line, font=self.status_font,
                      fill=status_color)
            y += 34
        if activity_lines:
            y += 12
            for line in activity_lines:
                draw.text((panel_left, y), line, font=self.label_font, fill=(251, 191, 36))
                y += 28
        if item_lines:
            y += 18
            for line in item_lines:
                draw.text((panel_left, y), line, font=self.label_font, fill=(226, 232, 240))
                y += 28
        draw.text((panel_left, canvas_height - 58), "s: 보호 요청   f: 보호 중지",
                  font=self.label_font, fill=(148, 163, 184))
        draw.text((panel_left, canvas_height - 30), "q: 프로그램 종료",
                  font=self.label_font, fill=(148, 163, 184))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
