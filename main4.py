import cv2
import numpy as np
import argparse

def is_snow(flake):
    # 눈 결정체의 예상 면적 범위
    min_area = 25
    max_area = 1000
    area = cv2.contourArea(flake)
    return min_area <= area <= max_area

# argparse를 사용하여 명령줄 인자를 받아옴
parser = argparse.ArgumentParser(description='Video source selection for snow detection')
parser.add_argument('--source', type=str, required=True, help='Video source: file path, rtsp, or other video stream URL')

args = parser.parse_args()

video_source = args.source
cap = cv2.VideoCapture(video_source)

# 출력 파일 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'X264'
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

_, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

count = 1
snow_count_sum = 0
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

    snow_count = 0  # 눈으로 감지된 객체의 수를 저장할 변수

    for contour in contours:
        if is_snow(contour):
            snow_count += 1  # 조건에 맞는 객체를 카운트
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    snow_count_sum += snow_count
    
    cv2.putText(frame2, f'Snow count: {snow_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame2, f'Snow count: {int(snow_count_sum/count)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 214, 0), 2)
    cv2.imshow('frame', frame2)
    out.write(frame2)
    
    count += 1

    if cv2.waitKey(1) == ord('q'):
        break

    gray1 = gray2

cap.release()
out.release()
cv2.destroyAllWindows()
