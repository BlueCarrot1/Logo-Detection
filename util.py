import requests
from PIL import Image
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