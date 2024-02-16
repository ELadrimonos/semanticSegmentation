from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import gradio as gr
import numpy as np

# Cargar el procesador y el modelo
processor = SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes')
model = AutoModelForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')


def segmentar_imagen(image):
    global model, processor
    image = Image.fromarray(image)
    # Procesar la imagen
    inputs = processor(images=image, return_tensors='pt')

    # Realizar la inferencia
    model = model.eval()
    outputs = model(**inputs)

    # Obtener los logits y realizar la interpolación
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode='bilinear',
        align_corners=False
    )

    # Obtener la segmentación predicha
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    colormap = plt.get_cmap('tab20').colors
    colormap = np.array(colormap) * 255
    colored_seg = np.zeros_like((*pred_seg.shape, 3), dtype=np.uint8)
    for label in range(18):
        # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        colored_seg[pred_seg == label] = colormap[label]
    return colored_seg


gr_image_input = gr.Image(image_mode='RGB', type='numpy')
gr_image_output = gr.Image(type='numpy', label='Segmented Image')

demo = gr.Interface(fn=segmentar_imagen, inputs=gr_image_input, outputs=gr_image_output)
demo.launch()
