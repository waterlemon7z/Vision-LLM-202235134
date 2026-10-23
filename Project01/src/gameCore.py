import cv2
import random
from pathlib import Path


class BasketGame:
    """
    - name        : BasketGame
    - description : 바구니 게임을 정의하는 클래스
    :return:
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.hand_loc: tuple | None = None
        self.hand_pinch = False
        self.hand_dist = 0.0
        self.finger_thumb = None
        self.finger_index = None

        self.score = 0

        root = Path(__file__).resolve().parent.parent

        # 이미지 로드 (흰색 배경 투명화)
        self.basket = self.load_image(root / "assets/basket.jpg")
        self.chicken = self.load_image(root / "assets/chicken.png")

        self.basket_h, self.basket_w = self.basket.shape[:2]

        # 바구니 크기
        self.min_basket_ratio = 0.10
        self.max_basket_ratio = 0.40
        self.distance_scale = 0.25

        # 닭 크기 (화면 너비의 8%)
        self.chicken_width_ratio = 0.20

        cw = max(1, int(width * self.chicken_width_ratio))
        ch = max(
            1,
            int(cw * self.chicken.shape[0] / self.chicken.shape[1])
        )
        self.chicken = cv2.resize(self.chicken, (cw, ch))

        # 자유낙하 설정 (화면 높이 기준)
        self.gravity = 0.9  # 화면 높이 / s²

        self.chicken_x = 0.0
        self.chicken_y = 0.0
        self.chicken_vy = 0.0
        self.chicken_active = False

        self.respawn_delay = 0.5
        self.respawn_timer = 0.0

    @staticmethod
    def load_image(path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(path)

        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # 흰색 배경 제거
        white_mask = cv2.inRange(
            image[:, :, :3],
            (240, 240, 240),
            (255, 255, 255)
        )

        image[:, :, 3] = cv2.bitwise_and(
            image[:, :, 3],
            cv2.bitwise_not(white_mask)
        )

        return image

    def start(self):
        self.score = 0
        self.spawn_chicken()

    def spawn_chicken(self):
        """
        - name        : spawn_chicken
        - description : 화면 위쪽의 랜덤 X 위치에서 생성
        :return:
        """
        self.chicken_x = random.uniform(0.1, 0.9)
        self.chicken_y = 0.0
        self.chicken_vy = 0.0
        self.chicken_active = True

    def update_cv_input(self, hand_input):
        """
        - name        : update_cv_input
        - description : 손 모양에 대한 정보 게임 인스턴스에 업데이트
        :param hand_input:
        :return none:
        """
        if hand_input["detected"]:
            self.hand_loc = hand_input["center"]
            self.hand_dist = hand_input["distance"]
            self.finger_thumb = hand_input["finger_thumb"]
            self.finger_index = hand_input["finger_index"]
            self.hand_pinch = hand_input["pinching"]
        else:
            self.hand_loc = None
            self.hand_dist = 0.0
            self.finger_thumb = None
            self.finger_index = None
            self.hand_pinch = False

    @staticmethod
    def intersects(a, b, c, d):
        """
        - name        : intersects
        - description : 선분 AB와 선분 CD가 교차하는지 검사.
                        AB: 닭의 이전 위치 → 현재 위치
                        CD: 엄지 끝 → 검지 끝
        :return boolean
        """

        rx, ry = b[0] - a[0], b[1] - a[1]
        sx, sy = d[0] - c[0], d[1] - c[1]
        # 벡터의 외적으로 계산
        denominator = rx * sy - ry * sx

        if abs(denominator) < 1e-8:
            return False

        qx, qy = c[0] - a[0], c[1] - a[1]

        t = (qx * sy - qy * sx) / denominator
        u = (qx * ry - qy * rx) / denominator
        # 실제 선분끼리 겹치는가
        return 0 <= t <= 1 and 0 <= u <= 1

    def update(self, dt):
        """
        - name        : update
        - description : 게임을 시간에 따라 업데이트
        :param dt:
        :return:
        """
        dt = max(0.0, min(dt, 0.1)) # dt는 이전 업데이트 기준 몇초가 흘렀는가

        if not self.chicken_active:
            self.respawn_timer -= dt # 치킨이 없고 다시 스폰될 시간이면

            if self.respawn_timer <= 0:
                self.spawn_chicken()

            return

        # 이전 위치 보관
        previous_y = self.chicken_y

        # 자유낙하: y = y0 + v0*t + 0.5*g*t²
        self.chicken_y += (
            self.chicken_vy * dt
            + 0.5 * self.gravity * dt * dt
        )
        self.chicken_vy += self.gravity * dt

        # 정규화 좌표 → 픽셀 좌표
        x = self.chicken_x * self.width
        old_y = previous_y * self.height
        new_y = self.chicken_y * self.height

        # 손이 있을 때만 충돌 판정
        if self.finger_thumb is not None and self.finger_index is not None:
            caught = self.intersects(
                (x, old_y),
                (x, new_y),
                self.finger_thumb,
                self.finger_index
            )

            if caught:
                self.score += 1
                self.chicken_active = False
                self.respawn_timer = self.respawn_delay
                return

        # 닭이 화면 아래로 완전히 떨어지면 재생성
        bottom = 1.0 + self.chicken.shape[0] / (2 * self.height)

        if self.chicken_y > bottom:
            self.chicken_active = False
            self.respawn_timer = self.respawn_delay

    @staticmethod
    def draw_image(frame, image, x, y):
        """
        - name        : draw_image
        - description : BGRA 이미지를 중심 좌표 기준으로 합성
        :return:
        """
        ih, iw = image.shape[:2]
        fh, fw = frame.shape[:2]

        x1 = int(x - iw // 2)
        y1 = int(y - ih // 2)
        x2, y2 = x1 + iw, y1 + ih

        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(fw, x2), min(fh, y2)

        if fx1 >= fx2 or fy1 >= fy2:
            return

        bx1, by1 = fx1 - x1, fy1 - y1
        bx2 = bx1 + (fx2 - fx1)
        by2 = by1 + (fy2 - fy1)

        overlay = image[by1:by2, bx1:bx2]
        background = frame[fy1:fy2, fx1:fx2]

        alpha = overlay[:, :, 3:4] / 255.0

        frame[fy1:fy2, fx1:fx2] = (
            overlay[:, :, :3] * alpha
            + background * (1 - alpha)
        ).astype("uint8")

    def draw_display(self, frame):
        """
        - name        : draw_display
        - description : draw_image() 호출해서 각각 이미지 삽입
        :return:
        """
        fh, fw = frame.shape[:2]

        # 닭 그리기
        if self.chicken_active:
            self.draw_image(
                frame,
                self.chicken,
                self.chicken_x * fw,
                self.chicken_y * fh
            )

        # 바구니 그리기
        if self.hand_loc is not None:
            basket_ratio = self.hand_dist * self.distance_scale

            basket_ratio = max(
                self.min_basket_ratio,
                min(self.max_basket_ratio, basket_ratio)
            )

            bw = max(1, int(fw * basket_ratio))
            bh = max(
                1,
                int(bw * self.basket_h / self.basket_w)
            )

            basket = cv2.resize(self.basket, (bw, bh))

            self.draw_image(
                frame,
                basket,
                *self.hand_loc
            )

        # 점수 표시
        cv2.putText(
            frame,
            f"SCORE: {self.score}",
            (int(fw * 0.03), int(fh * 0.08)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, fw / 900),
            (0, 255, 255),
            2
        )

        return frame
