import asyncio
from fastapi import FastAPI, HTTPException, WebSocket
from contextlib import asynccontextmanager
import serial
import serial.tools.list_ports
import threading
import time
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

selected_port = None
ser = None
latest_data = None
running = True

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

origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# WebSocket 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        if latest_data:
            await websocket.send_json(latest_data)
        await asyncio.sleep(2)