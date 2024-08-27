from datetime import datetime, timedelta, timezone

from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
import cv2
import numpy as np
import uvicorn
import io
import time
import main1
from fastapi.background import BackgroundTasks
import urllib.parse
import sqlite3
from datetime import datetime

# 데이터베이스 연결 및 테이블 생성
conn = sqlite3.connect('snow_count.db')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS SnowAverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    avg_snow_count INTEGER NOT NULL
)
''')

conn.commit()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_frames(video_source):
    camera = cv2.VideoCapture(1)  # 비디오 파일 또는 카메라
    if isinstance(video_source, str):
        fps = camera.get(cv2.CAP_PROP_FPS)  # 비디오의 FPS를 가져옴
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if isinstance(video_source, str):
                time.sleep(1/fps)  # FPS에 맞춰 대기
    camera.release()

def is_snow(flake):
    # 눈 결정체의 예상 면적 범위
    min_area = 25
    max_area = 1000
    area = cv2.contourArea(flake)
    color = cv2.drawContours
    return min_area <= area <= max_area

def generate_frames2(video_source, interval_minutes=1):
    cap = cv2.VideoCapture(video_source)  # 카메라 장치 번호

    _, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    count = 1
    snow_count_sum = 0
    interval_start_time = time.time()  # N분 간의 시작 시간

    while True:
        _, frame2 = cap.read()
        if frame2 is None:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        snow_count = 0

        for contour in contours:
            if is_snow(contour):
                snow_count += 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        snow_count_sum += snow_count
        
        avg_snow_count = int(snow_count_sum / count)
        
        cv2.putText(frame2, f'Snow count: {snow_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame2, f'Avg Snow count: {avg_snow_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 214, 0), 2)
        
        count += 1

        # N분이 지났는지 확인하고 평균 값 저장
        if time.time() - interval_start_time >= interval_minutes * 60:
            save_avg_to_db(avg_snow_count)
            snow_count_sum = 0  # 다음 N분을 위해 초기화
            count = 1
            interval_start_time = time.time()

        # frame2를 JPEG로 인코딩하고 바이트 스트림으로 변환
        ret, buffer = cv2.imencode('.jpg', frame2)
        frame = buffer.tobytes()
        
        # 스트리밍 포맷에 맞춰 frame 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        gray1 = gray2

    cap.release()

def save_avg_to_db(avg_snow_count):
    timestamp = datetime.now().isoformat()
    cursor.execute("INSERT INTO SnowAverage (timestamp, avg_snow_count) VALUES (?, ?)", (timestamp, avg_snow_count))
    conn.commit()

# def generate_frames2(video_source):
#     cap = cv2.VideoCapture(video_source)  # 카메라 장치 번호

#     _, frame1 = cap.read()
#     gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

#     count = 1
#     snow_count_sum = 0
#     while True:
#         _, frame2 = cap.read()
#         if frame2 is None:
#             break
#         gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#         diff = cv2.absdiff(gray1, gray2)
#         _, thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
#         kernel = np.ones((3, 3), np.uint8)
#         dilated = cv2.dilate(thresh, kernel, iterations=2)

#         contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         snow_count = 0

#         for contour in contours:
#             if is_snow(contour):
#                 snow_count += 1
#                 x, y, w, h = cv2.boundingRect(contour)
#                 cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         snow_count_sum += snow_count
        
#         cv2.putText(frame2, f'Snow count: {snow_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.putText(frame2, f'Avg Snow count: {int(snow_count_sum/count)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 214, 0), 2)
        
#         count += 1

#         # frame2를 JPEG로 인코딩하고 바이트 스트림으로 변환
#         ret, buffer = cv2.imencode('.jpg', frame2)
#         frame = buffer.tobytes()
        
#         # 스트리밍 포맷에 맞춰 frame 전송
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
#         gray1 = gray2

#     cap.release()

def generate_original_stream(video_source):
    print(video_source)
    cap = cv2.VideoCapture(video_source)
    if isinstance(video_source, str):
        fps = cap.get(cv2.CAP_PROP_FPS)
    _, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        _, frame2 = cap.read()
        if frame2 is None:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if is_snow(contour):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame2)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        gray1 = gray2  # 다음 프레임 비교를 위해 현재 프레임을 저장
        if isinstance(video_source, str):
            time.sleep(1/fps)  # FPS에 맞춰 대기

    cap.release()

def generate_thresh_stream(video_source):
    cap = cv2.VideoCapture(video_source)
    if isinstance(video_source, str):
        fps = cap.get(cv2.CAP_PROP_FPS)
    _, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        _, frame2 = cap.read()
        if frame2 is None:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)

        ret, buffer = cv2.imencode('.jpg', thresh)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        gray1 = gray2  # 다음 프레임 비교를 위해 현재 프레임을 저장
        if isinstance(video_source, str):
            time.sleep(1/fps)

    cap.release()

def stop_camera():
    # 카메라 사용 중지를 위한 코드 작성
    # 예: cv2.VideoCapture 객체 해제 등
    pass

@app.get("/video/original")
async def video_feed_1(source: str = Query(..., description="Video source URL or file path")):
    decoded_source = urllib.parse.unquote(source)
    return StreamingResponse(
        generate_original_stream(decoded_source),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/video/modified")
async def video_feed_2(source: str = Query(..., description="Video source URL or file path")):
    decoded_source = urllib.parse.unquote(source)
    return StreamingResponse(
        generate_frames2(decoded_source), 
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get('/')
def index():
    return {"message": "Hello, world"}

if __name__=="__main__":
    config = uvicorn.Config('backend_snow:app', host='0.0.0.0')
    server = uvicorn.Server(config)
    server.run()