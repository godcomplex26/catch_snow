import cv2
import numpy as np

def is_snow(flake):
    min_area = 25
    max_area = 1000
    area = cv2.contourArea(flake)
    return min_area <= area <= max_area

def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

video_file = 'snow.mp4'
cap = cv2.VideoCapture(video_file)

_, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
last_centers = {}

while True:
    _, frame2 = cap.read()
    if frame2 is None:
        break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_centers = {}

    for contour in contours:
        if is_snow(contour):
            center = get_center(contour)
            if center is not None:
                x, y, w, h = cv2.boundingRect(contour)
                new_centers[center] = (x, y, w, h)

                if center in last_centers:
                    last_x, last_y, _, _ = last_centers[center]  # x, y, w, h 중 x와 y만 추출
                    if y > last_y:  # 현재 중심점이 이전보다 아래에 있으면
                        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)


    last_centers = new_centers
    cv2.imshow('frame', frame2)

    if cv2.waitKey(1) == ord('q'):
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()