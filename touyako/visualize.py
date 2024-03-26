from glob import glob
from os.path import join as opj
from tqdm import tqdm
from tabulate import tabulate
import random
import os
from collections import defaultdict
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from PIL import UnidentifiedImageError


def concat_images(image_anns, w=224, h=224, r=3, c=3):
    result = Image.new('RGB', (w * r, h * c))
    images = []
    for image_ann in image_anns:
        image_path = image_ann.strip().split('\t')[0]
        add_bbox = False
        if '#' in image_path:
            image_path, x, y, width, height = image_path.split('#')
            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)
            add_bbox = True
        try:
            image = Image.open(image_path)
        except (FileNotFoundError, UnidentifiedImageError):
            print('no', image_path)
            image = Image.new('RGB', (w, h))
        if add_bbox:
            draw = ImageDraw.Draw(image)
            draw.rectangle([x, y, x + width, y + height], outline='red', width=3)
        image = image.resize((w, h))
        images.append(image)
    for i in range(r):
        for j in range(c):
            try:
                image = images[i + j * 3]
            except IndexError:
                image = Image.new('RGB', (w, h))
            result.paste(im=image, box=(i * w, j * h))
    return result


def create_visualize_generater(image_anns, w=224, h=224, r=3, c=3):
    batch_size = r * c
    for index in range(0, len(image_anns), batch_size):
        result = concat_images(image_anns[index:index + batch_size], w, h, r, c)
        print(index)
        plt.figure(figsize=(16, 16))
        plt.imshow(result)
        yield
