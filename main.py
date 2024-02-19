from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from PIL import Image
from torch import nn

# Cargar el procesador y el modelo
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")


def segment_image(image):
    image = Image.fromarray(image)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    colormap = plt.get_cmap('tab20').colors  # 'tab20' has 20 distinct colors
    colormap = np.array(colormap) * 255  # Convert to 0-255 range

    colored_seg = np.zeros((*pred_seg.shape, 3), dtype=np.uint8)
    for label in range(18):  # 0 to 17 inclusive
        colored_seg[pred_seg == label] = colormap[label]
    return colored_seg


gr_image_input = gr.Image(image_mode='RGB', type='numpy')
gr_image_output = gr.Image(type='numpy', label='Segmented Image')

demo = gr.Interface(fn=segment_image, inputs=gr_image_input, outputs=gr_image_output)
demo.launch()
