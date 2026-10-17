"""
 - Filename    :  20261017_hw2.py
 - Author      :  MinSeok Choi
 - Homework topic :  Morphological operation을 적용하여 최종 binary mask 생성
                     Binary mask로 외곽선 검출
                     최종 농구공 영역 및 외곽선 선택
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

COLOR_BASKETBALL_BGR = (88, 129, 248)

if __name__ == '__main__':
    # Step 1 : cvt image bgr to lab
    basketball = cv2.imread('../src/images/basketball_crop.jpg')
    basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(basketball_lab)

    # Step 2 : Do CLAHE to reinforce lightness
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe_enhanced = clahe.apply(l)
    basketball_enhanced_lab = cv2.merge([l_clahe_enhanced, a, b])

    # Step 3 : Create binary mask by using reinforced image
    lower_basketball_lab = np.array([70, 130, 70])
    upper_basketball_lab = np.array([180, 230, 185])
    basketball_mask = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

    # Step 4 : Do Morphological Operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    basketball_mask_opened = cv2.morphologyEx(basketball_mask, cv2.MORPH_OPEN, kernel)

    # Step 5 : Extract contours
    final_basketball_img = basketball.copy()
    contours, _ = cv2.findContours(
        basketball_mask_opened,  # binary mask w/ morphology open
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 6 : Visualize the biggest contour
    if len(contours) > 0:
        # biggest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Fill contour w/ basketball color
        final_mask = np.stack([np.zeros_like(basketball_mask_opened)] * 3,axis=2) #np.stack for applying rgb color
        cv2.drawContours(final_mask,[largest_contour],-1, COLOR_BASKETBALL_BGR, thickness=cv2.FILLED)
        alpha = 0.35
        final_basketball_img = cv2.addWeighted(final_basketball_img, 1.0, final_mask, alpha, 0)

        # Drawing masking edge w/ basketball color
        cv2.drawContours(final_basketball_img, [largest_contour], -1, COLOR_BASKETBALL_BGR, 2)

    # Final step : save a result image and show.
    cv2.imwrite('./hw2_basketball_contour_rst.jpg', final_basketball_img)
    final_basketball_img_rgb = cv2.cvtColor(final_basketball_img, cv2.COLOR_BGR2RGB)
    plt.imshow(final_basketball_img_rgb)
    plt.axis("off")
    plt.title('Basketball contour example')
    plt.show()
    # GOOD :D