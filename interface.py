from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

session = ort.InferenceSession("./MobileNet.onnx")

classLabels = [
    "类别1", "类别10", "类别11", "类别12", "类别13",
    "类别14", "类别15", "类别2", "类别3", "类别4",
    "类别5", "类别6", "类别7", "类别8", "类别9"
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

        logits = outs[0][0]                 # 原始模型输出（logits）
        logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        # ----- Softmax 得到概率分布 -----
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)   # 概率数组，总和为 1

        # ----- 获取最高和第二高的索引及概率 -----
        idx_top1 = np.argmax(probs)
        prob_top1 = probs[idx_top1]

        # 方法：将最高概率位置置为 -1，再取 argmax
        probs_copy = probs.copy()
        probs_copy[idx_top1] = -1
        idx_top2 = np.argmax(probs_copy)
        prob_top2 = probs_copy[idx_top2]

        # 转换为百分比数值（保留两位小数）
        confidence_top1 = round(float(prob_top1) * 100, 2)
        confidence_top2 = round(float(prob_top2) * 100, 2)

        return {
            "code": 0,
            "data": {
                "label": classLabels[idx_top1] if idx_top1 < len(classLabels) else "未知",
                "confidence": confidence_top1,
                "second_label": classLabels[idx_top2] if idx_top2 < len(classLabels) else "未知",
                "second_confidence": confidence_top2
            }
        }
    except Exception as e:
        return {"code": 1, "msg": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)