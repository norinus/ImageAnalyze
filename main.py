from fastapi import FastAPI,UploadFile,File, Form,Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import io
import base64
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from jinja2 import Environment, FileSystemLoader, select_autoescape
import logging
app = FastAPI()

env = Environment(loader=FileSystemLoader("templates"),autoescape=select_autoescape(['html']),extensions=['jinja2.ext.do'])
templates = Jinja2Templates(directory="templates")
templates.env = env

##분석 모델
model = YOLO('yolo11x.pt')

##로그 출력
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_objects(image:Image):
    img = np.array(image)
    results = model(img)
    class_names = model.names

    #결과를 바운딩 박스와 정확로 이미지에 표시
    for result in results:
        boxes = result.boxes.xyxy
        confidence = result.boxes.conf
        class_ids = result.boxes.cls
        for box, confidence, class_id in zip(boxes, confidence, class_ids):
            ##탐지 결과 박스 좌표
            x1, y1, x2, y2 = map(int, box)
            ##탐지 결과 라벨 생성
            label = f'{class_names[int(class_id)]} {confidence:.2f}'
            ##탐지 결과 박스
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ##탐지 결과 박스 상단에 클래스명과 확률
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # 탐지 결과를 로그에 출력
            logging.info(f"Detected: {class_names[int(class_id)]} | Confidence: {confidence:.2f} | Box: ({x1}, {y1}, {x2}, {y2})")

    result_image = Image.fromarray(img)
    return result_image

##분석 결과값과 결과 매핑 된 이미지
class DetectionResult(BaseModel):
    message:str
    image:str

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect", response_model=DetectionResult)
async def detect(message:str=Form(...), image:UploadFile=File(...)):

    image = Image.open(io.BytesIO(await image.read()))

    if image.mode != "RGBA":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    result_image = detect_objects(image)
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return DetectionResult(message=message,image=img_str)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)