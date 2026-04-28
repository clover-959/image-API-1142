from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import os
import urllib.request

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ====================== 🔥 在这里填你的模型下载链接 ======================
MODEL_URL = "https://airportal.cn/192961/ArXEoY9xMk"
MODEL_PATH = "./MobileNet.onnx"

# 自动下载模型（第一次部署会自动下）
if not os.path.exists(MODEL_PATH):
    print("正在下载模型...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("模型下载完成！")
    except Exception as e:
        print("下载失败", e)

# 加载模型
session = ort.InferenceSession(MODEL_PATH)

classLabels = [
    "类别1", "类别2", "类别3", "类别4", "类别5",
    "类别6", "类别7", "类别8", "类别9", "类别10",
    "类别11", "类别12", "类别13", "类别14", "类别15"
]


def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        tensor = process_image(img_bytes)
        inputs = {session.get_inputs()[0].name: tensor}
        outs = session.run(None, inputs)

        preds = outs[0][0]
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

        # 真实 Softmax 置信度
        exp_preds = np.exp(preds - np.max(preds))
        softmax = exp_preds / np.sum(exp_preds)

        idx = np.argmax(softmax)
        true_conf = float(softmax[idx])

        return {
            "code": 0,
            "data": {
                "label": classLabels[idx] if idx < len(classLabels) else "未知",
                "confidence": round(true_conf * 100, 2)
            }
        }
    except Exception as e:
        return {"code": 1, "msg": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10000)