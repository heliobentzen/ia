# Aula 53 — Visão Computacional

> **Módulo 11 · Aplicações de ML em Problemas Reais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Aplicar classificação, detecção de objetos e segmentação
- Usar YOLO para detecção em tempo real
- Compreender as tarefas de visão e seus casos de uso

---

## 1. Tarefas de Visão Computacional

| Tarefa | Saída | Modelos |
|--------|-------|---------|
| Classificação | Classe da imagem | ResNet, EfficientNet, ViT |
| Detecção | BBoxes + classes | YOLO, DETR, Faster R-CNN |
| Segmentação semântica | Pixel-level class | U-Net, DeepLab |
| Segmentação instância | Objeto-level masks | Mask R-CNN, SAM |
| Estimativa de pose | Keypoints | MediaPipe, OpenPose |
| OCR | Texto na imagem | Tesseract, PaddleOCR |

---

## 2. Classificação com Transfer Learning

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Usar MobileNetV3 — leve e preciso
base = tf.keras.applications.MobileNetV3Small(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"Total params: {model.count_params():,}")

# Inferência em uma imagem
def classify_image(image_path, model, preprocess_fn, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_fn(img_array[np.newaxis])
    
    preds = model.predict(img_array, verbose=0)
    top5 = preds[0].argsort()[-5:][::-1]
    
    for idx in top5:
        print(f"  {class_names[idx]:30s}: {preds[0][idx]*100:.1f}%")
```

---

## 3. Detecção com YOLO

```python
# pip install ultralytics
from ultralytics import YOLO
import cv2
import numpy as np

# Carregar YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')  # nano = mais rápido

# Detecção em imagem
results = model('foto.jpg', conf=0.5, iou=0.45)

# Plotar resultados
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = model.names[cls]
        print(f"  {class_name}: {conf:.2f} @ ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

# Salvar imagem anotada
annotated = results[0].plot()
cv2.imwrite('resultado.jpg', annotated)

# Detecção em vídeo/webcam
# results = model.predict(source=0, stream=True)  # webcam em tempo real
```

---

## 4. Segmentação com SAM (Segment Anything Model)

```python
# Meta AI SAM — segmenta qualquer objeto com prompt de ponto/bbox
from segment_anything import SamPredictor, sam_model_registry
import numpy as np

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
predictor = SamPredictor(sam)

# Carregar imagem
import cv2
image = cv2.imread("imagem.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# Segmentar a partir de um ponto
input_point = np.array([[500, 375]])  # ponto na imagem
input_label = np.array([1])  # 1 = objeto de interesse

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)
print(f"Máscaras geradas: {masks.shape}, Scores: {scores}")
```

---

## Questões para Reflexão
1. Qual a diferença entre segmentação semântica e segmentação por instância?
2. Por que YOLO é preferido para aplicações em tempo real?
3. Como o SAM diferencia dos modelos de segmentação tradicionais?

## Referências
- Géron, cap. 14
- Documentação ultralytics.com, segment-anything.com

---
*Próxima aula → [Aula 54: Séries Temporais](aula-54-series-temporais.md)*
