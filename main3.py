import cv2
import numpy as np

def is_snow(flake, color_img):
    min_area = 25
    max_area = 1000
    min_rgb_threshold = (130, 130, 130)

    # 면적 조건 검사
    area = cv2.contourArea(flake)
    if area < min_area or area > max_area:
        return False

    # 마스크 생성 및 평균 RGB 값 계산
    mask = np.zeros(color_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [flake], -1, 255, -1)
    mean_val = cv2.mean(color_img, mask=mask)

    # RGB 평균값이 임계값 이상인지 검사
    if mean_val[0] < min_rgb_threshold[0] or mean_val[1] < min_rgb_threshold[1] or mean_val[2] < min_rgb_threshold[2]:
        return False

    return True

video_file = 'snow2.mp4'
cap = cv2.VideoCapture(video_file)

_, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    _, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if is_snow(contour, frame2):  # 컬러 이미지 사용
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame2)

    if cv2.waitKey(1) == ord('q'):
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()