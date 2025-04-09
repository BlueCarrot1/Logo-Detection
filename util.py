import requests
from PIL import Image,ImageDraw,ImageFont
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches 


def url_image(image):
    if os.path.exists(image):
        img = Image.open(image)
    else:
        img = Image.open(requests.get(image, stream=True).raw)
    return img


def plot_bbox(image, box_list, text = "logo"):
    fig, ax = plt.subplots()  
    ax.imshow(image)        
    for ind, bbox in enumerate(box_list):  
        x1, y1, x2, y2 = bbox 
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, text + str(ind), color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
    ax.axis('off')
    plt.show()


def plot_all_logo(image, box_list, save_path):
    """
    在原图上绘制box和文字，并保存修改后的图片。
    参数:
    - image: PIL.Image对象或图片路径。
    - box_list: 包含box信息和颜色指标的字典列表。
    - save_path: 保存修改后图片的路径。
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(20)
    for box in box_list:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    image.save(save_path)
    return 




