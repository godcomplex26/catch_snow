import cv2
import numpy as np

def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def is_snow(flake):
    min_area = 64
    max_area = 1000
    area = cv2.contourArea(flake)
    return min_area <= area <= max_area

video_file = 'snow.mp4'
cap = cv2.VideoCapture(video_file)
cap.set(cv2.CAP_PROP_FPS, 5)

_, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

count = 1
snow_count_sum = 0
last_centers = {}

while True:
    _, frame2 = cap.read()
    if frame2 is None:
        break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    snow_count = 0
    new_centers = {}

    for contour in contours:
        if is_snow(contour):
            center = get_center(contour)
            if center is not None:
                x, y, w, h = cv2.boundingRect(contour)
                new_centers[center] = y

                if center in last_centers and y > last_centers[center]:
                    snow_count += 1
                    cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    snow_count_sum += snow_count
    last_centers = new_centers

    cv2.putText(frame2, f'Snow count: {snow_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame2, f'Average snow count: {int(snow_count_sum / count)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 214, 0), 2)
    cv2.imshow('frame', frame2)
    
    count += 1

    if cv2.waitKey(1) == ord('q'):
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()
