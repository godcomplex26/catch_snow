import base64

# PNG 파일을 읽고 Base64로 인코딩하는 함수
def png_to_base64(png_path):
    with open(png_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")  # 텍스트 형태로 반환

# 사용 예시: 32x32.png 파일을 Base64로 변환
encoded_image = png_to_base64("32x32.png")

# Base64로 인코딩된 텍스트 출력 (코드에 삽입하기 위해서 출력 가능)
print(encoded_image)