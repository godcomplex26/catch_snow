from datetime import datetime, timedelta, timezone

from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
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

video_source = "rtsp://210.99.70.120:1935/live/cctv001.stream" # 1
# video_source = "snow.mp4"


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "e486f376246887e31a10485daa8df3f2ce45d7f178fbf178c8c4921ac084b47a"
ALGORITHM = "HS256"

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# first 'static' specify route path, second 'static' specify html files directory.
# app.mount('/static', StaticFiles(directory='static',html=True))
# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0')
    

# def generate_frames():
#     # 카메라 캡처 초기화
#     camera = cv2.VideoCapture('snow.mp4')  # 0은 기본 카메라
#     while True:
#         success, frame = camera.read()  # 프레임 캡처
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)  # 프레임을 JPEG로 인코딩
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 멀티파트 스트림 형식
#     camera.release()

def generate_frames():
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

def generate_frames2():
    cap = cv2.VideoCapture(1)  # 카메라 장치 번호

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
        
        cv2.putText(frame2, f'Snow count: {snow_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame2, f'Avg Snow count: {int(snow_count_sum/count)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 214, 0), 2)
        
        count += 1

        # frame2를 JPEG로 인코딩하고 바이트 스트림으로 변환
        ret, buffer = cv2.imencode('.jpg', frame2)
        frame = buffer.tobytes()
        
        # 스트리밍 포맷에 맞춰 frame 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        gray1 = gray2

    cap.release()

def generate_original_stream():
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

def generate_thresh_stream():
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


async def verify_token(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        expiration = payload.get("exp")
        if expiration is None:
            raise credentials_exception
        expire = datetime.fromtimestamp(expiration, tz=timezone.utc)
        if datetime.now(tz=timezone.utc) > expire:
            raise HTTPException(status_code=400, detail="Token expired")
        # 추가적인 토큰 검증 로직이 필요하면 여기에 구현합니다.
    except JWTError:
        raise credentials_exception
    return payload

def stop_camera():
    # 카메라 사용 중지를 위한 코드 작성
    # 예: cv2.VideoCapture 객체 해제 등
    pass

@app.get("/video/1")
async def video_feed_1():
    return StreamingResponse(
        generate_original_stream(),
        media_type='multipart/x-mixed-replace; boundary=frame'
        )

@app.get("/video/2")
async def video_feed_2():
    return StreamingResponse(generate_thresh_stream(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/')
def index():
    return {"message": "Hello, world"}


@app.get('/video')
def video_feed():
    return StreamingResponse(generate_frames2(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/test2", dependencies=[Depends(verify_token)])
async def read_items():
    return {"token_data": "Test!"}


@app.get("/logo")
def get_logo():
    return FileResponse("logo.png")
# from fastapi import FastAPI, Response
# from fastapi.responses import StreamingResponse
# import aiofiles
# from fastapi.staticfiles import StaticFiles
# import uvicorn

# app = FastAPI()

# # first 'static' specify route path, second 'static' specify html files directory.
# app.mount('/static', StaticFiles(directory='static',html=True))
# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0')

# @app.get("/video/")
# async def video_stream():
#     file_path = "output.mp4"  # 비디오 파일 경로
#     return StreamingResponse(file_iterator(file_path), media_type="video/mp4")

# async def file_iterator(file_path: str, chunk_size: int = 10 * 1024):
#     async with aiofiles.open(file_path, mode="rb") as f:
#         chunk = await f.read(chunk_size)
#         while chunk:
#             yield chunk
#             chunk = await f.read(chunk_size)


    
    

# CHUNK_SIZE = 1024*1024
# video_path = Path("snow2.mp4")


# @app.get("/video2")
# async def video_endpoint(range: str = Header(None)):
#     start, end = range.replace("bytes=", "").split("-")
#     start = int(start)
#     end = int(end) if end else start + CHUNK_SIZE
#     with open(video_path, "rb") as video:
#         video.seek(start)
#         data = video.read(end - start)
#         filesize = str(video_path.stat().st_size)
#         headers = {
#             'Content-Range': f'bytes {str(start)}-{str(end)}/{filesize}',
#             'Accept-Ranges': 'bytes'
#         }
#         return Response(data, status_code=206, headers=headers, media_type="video/mp4")

if __name__=="__main__":
    config = uvicorn.Config('main:app', host='0.0.0.0')
    server = uvicorn.Server(config)
    server.run()