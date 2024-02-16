from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

# Cargar el procesador y el modelo
processor = SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes')
model = AutoModelForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')

# Cargar la imagen
url = 'https://th.bing.com/th/id/OIP.I6Bn9EKsT2X-fBYMwohZXAHaLH?rs=1&pid=ImgDetMain'
image = Image.open(requests.get(url, stream=True).raw)

# Procesar la imagen
inputs = processor(images=image, return_tensors='pt', size=512)

# Realizar la inferencia
model = model.eval()
outputs = model(**inputs)

# Obtener los logits y realizar la interpolación
logits = outputs.logits
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode='bilinear',
    align_corners=False
)

# Obtener la segmentación predicha
pred_seg = upsampled_logits.argmax(dim=1)[0]

# Visualizar la segmentación
plt.imshow(pred_seg)
plt.show()
