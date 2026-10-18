# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출


# TODO: 각 프레임 이미지를 전처리
# TODO: Color Segmentation으로 Binary Mask 생성
# TODO: Morphological Operation을 적용하여 객체 영역 추출
# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay



import cv2
import numpy as np
from numpy.linalg import inv
COLOR_BALL_BGR = (88, 129, 248)



def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


def ballObjDetect(frame):
    # Kalman filter 변수 정의
    global found, A_t,G,R_t,C_t,Q_t,mu_t,sigma_t,radius
    ball = frame.copy()
    ball = cv2.GaussianBlur(ball, (7,7),0)
    ball_lab = cv2.cvtColor(ball, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(ball_lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe_enhanced = clahe.apply(l)
    ball_lab_enhanced = cv2.merge([l_clahe_enhanced, a, b])
    
    ball_green_lower = np.array([30, 60, 90], dtype=np.uint8)
    ball_green_upper = np.array([230, 115, 180], dtype=np.uint8)
    ball_mask = cv2.inRange(ball_lab_enhanced, ball_green_lower, ball_green_upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ball_mask_closed = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel,iterations=2)
    ball_mask_closed = cv2.morphologyEx(ball_mask_closed, cv2.MORPH_CLOSE, kernel,iterations=2)
    
    ball_obj_rst = frame.copy()
    contours, _ = cv2.findContours(
        ball_mask_closed,  # binary mask w/ morphology open
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        # biggest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Fill contour w/ basketball color
        final_mask = np.stack([np.zeros_like(ball_mask_closed)] * 3,axis=2) #np.stack for applying rgb color
        cv2.drawContours(final_mask,[largest_contour],-1, COLOR_BALL_BGR, thickness=cv2.FILLED)
        alpha = 0.35
        ball_obj_rst = cv2.addWeighted(ball_obj_rst, 1.0, final_mask, alpha, 0)

        # Drawing masking edge w/ basketball color
        cv2.drawContours(ball_obj_rst, [largest_contour], -1, COLOR_BALL_BGR, 2)
         # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)
    
        # 파란색 원
        cv2.circle(ball_obj_rst, center_circle, radius, (0,0,255), 3)
        cv2.circle(ball_obj_rst, center_circle, 5, (0,0,255), -1)
        
        
        # Moments를 통해 무게중심 (Centroid) 산출
        M = cv2.moments(largest_contour)
    
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
    
        centroid = (cX, cY)
    
        # 노란색 무게중심
        cv2.circle(ball_obj_rst, centroid, 5, (255,255,0), -1)
        if not found:
            found = True
    if found and (len(contours) > 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center_circle)]))
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]
    # 예측한 객체 위치에 노란 원 overlay
        cv2.circle(ball_obj_rst, (int(x_bel), int(y_bel)), int(radius),(0,255,255), 3)
        cv2.circle(ball_obj_rst, (int(x_bel), int(y_bel)), 5,(0,255,255), -1)

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contours) <= 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]
        # 예측한 객체 위치에 노란 원 overlay
        cv2.circle(ball_obj_rst, (int(x_bel), int(y_bel)), int(radius),(0,255,255), 3)
        cv2.circle(ball_obj_rst, (int(x_bel), int(y_bel)),5, (0,255,255), -1)
    
    
    return ball_obj_rst
    
if __name__ == '__main__':
    pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue leaky=downstream max-size-buffers=2 "
        "max-size-bytes=0 max-size-time=0 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    A_t = np.array([[1, 1], [0, 1]])
    G = np.array([[0.5], [1]])
    R_t = G.dot(G.transpose())
    C_t = np.array([[1, 0]])
    Q_t = np.array([[1]])
    mu_t = np.array([[0, 0], [0, 0]])
    sigma_t = np.array([[0, 0], [0, 0]])
    found =False
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        # ==================================
        ball = frame.copy()
        ball = cv2.GaussianBlur(ball, (7,7),0)
        ball_lab = cv2.cvtColor(ball, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(ball_lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe_enhanced = clahe.apply(l)
        ball_lab_enhanced = cv2.merge([l_clahe_enhanced, a, b])
        
        ball_green_lower = np.array([30, 60, 90], dtype=np.uint8)
        ball_green_upper = np.array([230, 115, 180], dtype=np.uint8)
        ball_mask = cv2.inRange(ball_lab_enhanced, ball_green_lower, ball_green_upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ball_mask_closed = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel,iterations=2)
        ball_mask_closed = cv2.morphologyEx(ball_mask_closed, cv2.MORPH_CLOSE, kernel,iterations=2)
        
        ball_obj_rst = frame.copy()
        contours, _ = cv2.findContours(
            ball_mask_closed,  # binary mask w/ morphology open
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
    
        if len(contours) > 0:
            # biggest contour
            largest_contour = max(contours, key=cv2.contourArea)
    
            # Fill contour w/ basketball color
            final_mask = np.stack([np.zeros_like(ball_mask_closed)] * 3,axis=2) #np.stack for applying rgb color
            cv2.drawContours(final_mask,[largest_contour],-1, COLOR_BALL_BGR, thickness=cv2.FILLED)
            alpha = 0.35
            ball_obj_rst = cv2.addWeighted(ball_obj_rst, 1.0, final_mask, alpha, 0)
    
            # Drawing masking edge w/ basketball color
            cv2.drawContours(ball_obj_rst, [largest_contour], -1, COLOR_BALL_BGR, 2)
                # 최소 외접원 중심 및 반지름
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center_circle = (int(x), int(y))
            radius = int(radius)
        
            # 파란색 원
            cv2.circle(ball_obj_rst, center_circle, radius, (0,0,255), 3)
            cv2.circle(ball_obj_rst, center_circle, 5, (0,0,255), -1)
            
            
            # Moments를 통해 무게중심 (Centroid) 산출
            M = cv2.moments(largest_contour)
        
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
        
            centroid = (cX, cY)
        
            # 노란색 무게중심
            cv2.circle(ball_obj_rst, centroid, 5, (255,255,0), -1)
            if not found:
                found = True
        if found and (len(contours) > 0):
            mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center_circle)]))
            x_bel, y_bel = mu_t[0][0], mu_t[0][1]
        # 예측한 객체 위치에 노란 원 overlay
            # cv2.circle(ball_obj_rst, (int(x_bel), int(y_bel)), int(radius),(0,255,255), 3)
            # cv2.circle(ball_obj_rst, (int(x_bel), int(y_bel)), 5,(0,255,255), -1)
    
        # 측정값 미사용 (occluded) -> Prediction
        elif found and (len(contours) <= 0):
            mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
            x_bel, y_bel = mu_t[0][0], mu_t[0][1]
            # 예측한 객체 위치에 노란 원 overlay
        if found:
            cv2.circle(ball_obj_rst, (int(x_bel), int(y_bel)), int(radius),(0,255,255), 3)
            cv2.circle(ball_obj_rst, (int(x_bel), int(y_bel)),5, (0,255,255), -1)
                    
        cv2.imshow("VideoCapture with GStreamer",cv2.flip(ball_obj_rst,1))

    cap.release()
    cv2.destroyAllWindows()
