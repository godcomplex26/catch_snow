import base64
from io import BytesIO
import os
import signal
import sys
import threading
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
import urllib.parse
import cv2
import sqlite3
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, Query, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import serial
import json
import serial.tools.list_ports
from pystray import Icon, Menu, MenuItem
from PIL import Image, ImageDraw
import requests
# from findport import list_serial_ports, select_serial_port

# Base64로 인코딩된 이미지 데이터를 복사해 이 변수에 저장
encoded_image = """
iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAANjSURBVHgB7VZBbtpQEJ0xUdVFq5AbGAXSdBXIoklUKZATND1ByAnqngBzAsgJQk6Q9ASYLkKyCXTVqhDhniBEyiKtwNP5tgH728YGlu2TwPb3/Hnvz5+Zb4B/HQgrgpr5NF8mPxOPusNF5i8lwCZVFI3vTvhRlV6bgGjA2KqyGDPO18IC6OtuGYhq4Kw4zn0Vi3f6XAtYhLy1q/N/xT9IJv8briCVV5/3M2AdD+8+w6oCOOxlUPDcQ2zws46HnZZkxyKgwsRlD01kJBIJsJ0q2ITpfpOOxW41Zo7Gc2rTAYuOOCcM2U6BJEgpIuxpXnWDXZfjyAWYrM629ZkPrITaQQKICMzLaLcUQS5Bp1pwAJOEtWhDtkkUAUG+eds/3rzpnwfIRWIq+CB+XCE1aR6Tiai5SEFRnh8rINvulXI3/QESXSKQ6iNv5ku+qiDSqJU/8Tmw4NvsfaBnzBfA5DpvUpNcYgI0JRM1ZJo8Znru04kFZK97ml1OMxjK2DqTzK4CExG6EsOO52mYSMB2eyAayowcqd7fzx313m/5nNt7bNFHDn3XaUik4WH3i1+QMmtMiF0I6A0B7/k5h70s7vnauN/fOoUl4PaPgfs4xGJnQ7YJREDtDNITcmYfjmkcW/ORcPqHq4auwkzWAgPPo/wkLoR0Ze6/Nb3vRW6gQp8IYchCq/d7b0Idc3nWmbTsPpq8mGoiARyTIhu7QJ9z0QvESUhCIYn9w0u109kwC4Vpctmlabdg8hxK3LojGtncMkTFn7XcB3Zkm9Sf1x/8A3byesk1bt0XEIG5AohQ9Qmw6Jdsg2gfx2HgDxMoMfkZzEFAgJcEwfKd7b2DrQYPOgcMJyjf6/13uZakuusQdzLyUR2GQBmKKlj7PXqYkIxePmW8e+zYdOyOJo/7fHAy9w9yBsQgEAGzkBFODVcei3lVCdoUhmHkufbPcvamd7n2PB7YyZwA4TngLxkte/1DgxiIc4PQ/mI6JrSGI2t0AQkQ+T2Qve1VWIg+04SNMaWq5kHGlIhL7plRmtry59j9XnY1Aa6IGjP7V4+i7zunIpdlnoWpkke9v5dL3D1jv4jkSESCE5YU6zSqMy4tQGC7/V0dY8oOc2DFImERjNGLp7OoqlhZgBeiBJHW18U94ePjMqT/4cVfZNp1ptgvGtYAAAAASUVORK5CYII=
"""


# 사용자 디렉토리에서 애플리케이션 데이터 디렉토리 경로 가져오기
app_data_dir = Path(os.getenv('APPDATA') if os.name == 'nt' else os.path.expanduser('~/.local/share')) / 'com.snowd'

app_data_dir.mkdir(parents=True, exist_ok=True)

db_path = app_data_dir / 'todos.db'

# 포트 정보 저장을 위한 변수
selected_port = None
ser = None
latest_data = None
running = True

origins = ["*"]

# 포트 선택을 위한 Pydantic 모델 (device 이름만 필요)
class PortSelection(BaseModel):
    device: str

def read_serial_data():
    global ser, latest_data, running
    while running:
        if ser:
            try:
                line = ser.readline().decode('utf-8').strip()
                latest_data = json.loads(line)
            except Exception as e:
                latest_data = None
                print(f"Error reading from serial port: {e}")
        time.sleep(1)  # 대기 시간을 조절


@asynccontextmanager
async def lifespan(app: FastAPI):
    global running
    running = True
    # 시리얼 데이터를 읽는 백그라운드 스레드 시작
    thread = threading.Thread(target=read_serial_data, daemon=True)
    thread.start()

    yield  # 애플리케이션이 실행되는 동안

    # 애플리케이션 종료 시점에서 실행될 코드
    running = False
    if ser and ser.is_open:
        ser.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def initialize_db():
    conn = sqlite3.connect(str(db_path))
    # conn = sqlite3.connect('snow_count.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS SnowAverage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        avg_snow_count INTEGER NOT NULL
    )
    ''')

    conn.commit()
    conn.close()
    
initialize_db()

# 데이터베이스 연결을 생성하는 함수
def get_db_connection():
    conn = sqlite3.connect(str(db_path))
    return conn

def is_snow(flake):
    min_area = 25
    max_area = 1000
    area = cv2.contourArea(flake)
    return min_area <= area <= max_area

def save_avg_to_db(avg_snow_count):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("INSERT INTO SnowAverage (timestamp, avg_snow_count) VALUES (?, ?)", (timestamp, avg_snow_count))
    conn.commit()
    conn.close()

def generate_original_stream(video_source):
    print(video_source)
    cap = cv2.VideoCapture(video_source)
    # if isinstance(video_source, str):
    #     fps = cap.get(cv2.CAP_PROP_FPS)
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
        # if isinstance(video_source, str):
        #     time.sleep(1/fps)  # FPS에 맞춰 대기

    cap.release()

def generate_frames1(video_source):
    cap = cv2.VideoCapture(video_source)
    
    if isinstance(video_source, str):
        fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오의 FPS를 가져옴
        frame_delay = 1 / fps  # 각 프레임 사이의 대기 시간 계산
    else:
        frame_delay = None
        
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if frame_delay:
            time.sleep(frame_delay)  # FPS에 맞춰 대기
    cap.release()

def generate_frames2(video_source, interval_minutes=0.1):
    cap = cv2.VideoCapture(video_source)

    if isinstance(video_source, str):
        fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오의 FPS를 가져옴
        frame_delay = 1 / fps  # 각 프레임 사이의 대기 시간 계산
    else:
        frame_delay = None

    _, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    count = 1
    snow_count_sum = 0
    interval_start_time = time.time()

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

        if time.time() - interval_start_time >= interval_minutes * 60:
            save_avg_to_db(avg_snow_count)
            snow_count_sum = 0
            count = 1
            interval_start_time = time.time()

        ret, buffer = cv2.imencode('.jpg', frame2)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        gray1 = gray2

        if frame_delay:
            time.sleep(frame_delay)  # FPS에 맞춰 대기

    cap.release()

def get_recent_avg(n_minutes):
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = datetime.now()
    time_threshold = current_time - timedelta(minutes=n_minutes)
    
    cursor.execute("SELECT avg_snow_count FROM SnowAverage WHERE timestamp >= ?", (time_threshold.isoformat(),))
    rows = cursor.fetchall()
    conn.close()
    
    if rows:
        avg_values = [row[0] for row in rows]
        overall_avg = sum(avg_values) / len(avg_values)
    else:
        overall_avg = 0
    
    return overall_avg

@app.get("/ports")
def get_serial_ports():
    ports = serial.tools.list_ports.comports()
    port_list = []
    
    for port in ports:
        port_info = {
            "device": port.device,
            "name": port.name,
            "description": port.description,
            "hwid": port.hwid
        }
        port_list.append(port_info)
    
    if not port_list:
        raise HTTPException(status_code=404, detail="No serial ports found.")
    
    return port_list

@app.post("/select_port")
def select_serial_port(port_selection: PortSelection):
    global selected_port, ser
    
    available_ports = serial.tools.list_ports.comports()
    for port in available_ports:
        if port.device == port_selection.device:
            selected_port = {
                "device": port.device,
                "name": port.name,
                "description": port.description,
                "hwid": port.hwid
            }
            try:
                ser = serial.Serial(selected_port['device'], 9600, timeout=2)
                time.sleep(2)  # 아두이노 리셋 시간 대기
            except serial.SerialException as e:
                raise HTTPException(status_code=500, detail=f"Could not open serial port: {str(e)}")
            return {"message": "Port selected successfully", "selected_port": selected_port}
    
    raise HTTPException(status_code=404, detail="Selected port not found.")

@app.get("/selected_port")
def get_selected_port():
    if selected_port is None:
        raise HTTPException(status_code=404, detail="No port has been selected.")
    return selected_port

@app.get("/dht22_data")
def get_dht22_data():
    if latest_data is None:
        raise HTTPException(status_code=404, detail="No data available.")
    return latest_data

@app.post("/restart")
def restart_server():
    """
    서버를 재시작하는 엔드포인트
    """
    def restart():
        python = sys.executable
        os.execl(python, python, *sys.argv)

    signal.signal(signal.SIGTERM, restart)  # SIGTERM 신호에 대해 restart 함수를 연결
    os.kill(os.getpid(), signal.SIGTERM)  # 현재 프로세스에 SIGTERM 신호를 보냄

    return JSONResponse(content={"message": "Server is restarting..."})

@app.get("/recent_avg")
def recent_avg(m: int = Query(5, description="The number of minutes to calculate the average over")):
    avg_value = get_recent_avg(m)
    return JSONResponse(content={"average_snow_count": avg_value, "minutes": m})

@app.get("/video/original")
async def video_feed_1(source: str = Query(..., description="Video source URL or file path")):
    decoded_source = urllib.parse.unquote(source)
    return StreamingResponse(
        generate_frames1(decoded_source),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/video/modified")
async def video_feed_2(source: str = Query(..., description="Video source URL or file path")):
    decoded_source = urllib.parse.unquote(source)
    return StreamingResponse(
        generate_frames2(decoded_source),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
    
# FastAPI POST 요청을 통해 서버 종료 명령을 받는 엔드포인트
@app.post("/shutdown")
def shutdown_server():
    """
    FastAPI 앱을 종료하는 엔드포인트
    """
    def shutdown():
        os.kill(os.getpid(), signal.SIGTERM)  # 현재 프로세스에 SIGTERM 신호를 보냄
    
    threading.Thread(target=shutdown).start()  # 별도의 스레드에서 종료 명령 실행
    return JSONResponse(content={"message": "Server is shutting down..."})

# def main():
#     # 현재 실행 중인 Python 파일의 이름을 감지합니다.
#     current_file = os.path.basename(__file__)
    
#     # 파일 확장자를 제거하고 모듈 이름으로 사용합니다.

#     # Uvicorn 서버를 해당 모듈 이름으로 실행합니다.
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_config=None)

# 아이콘에 사용할 이미지를 생성하는 함수
def create_image(width, height, color1, color2):
    # 빈 이미지 생성
    image = Image.new('RGB', (width, height), color1)
    dc = ImageDraw.Draw(image)

    # 이미지에 간단한 모양 그리기
    dc.rectangle(
        (width // 4, height // 4, width * 3 // 4, height * 3 // 4),
        fill=color2)

    return image

# 트레이 메뉴에서 사용할 종료 함수
def on_quit(icon, item):
    # FastAPI 서버 종료 요청을 보냄
    requests.post("http://127.0.0.1:8000/shutdown")  # FastAPI 서버에 종료 요청
    icon.stop()  # 트레이 아이콘 종료
    # sys.exit()

# Base64 문자열을 디코딩하여 이미지로 변환하는 함수
def base64_to_image(encoded_image):
    image_data = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(image_data))
    return image

# 시스템 트레이 아이콘 설정 함수
def setup_tray():
    image = base64_to_image(encoded_image)
    # image = create_image(64, 64, 'black', 'white')
    menu = Menu(MenuItem('Exit', on_quit))
    icon = Icon("icon", image, "snow-detector", menu)
    icon.run()

if __name__=="__main__":
    tray_thread = threading.Thread(target=setup_tray, daemon=True)
    tray_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
    # run_fastapi()
    # target 함수로 실행할 uvicorn.run을 전달하고, args로 함수의 파라미터를 넘김
    # api_thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8000}, daemon=False)
    # api_thread.start()
    # setup_tray()