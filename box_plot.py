import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIRS = {
    'english': '/home/wajeeha/Documents/classifier2/Dataset/english',
    'chinese': '/home/wajeeha/Documents/classifier2/Dataset/chinese',
    'hebrew': '/home/wajeeha/Documents/classifier2/Dataset/hebrew',
    'russian': '/home/wajeeha/Documents/classifier2/Dataset/hindi',
    'urdu': '/home/wajeeha/Documents/classifier2/Dataset/urdu'
}
IMAGE_SIZE = (32, 32)

def auto_crop(image):
    np_img = np.array(image)
    mask = np_img < 250
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = image.crop((x0, y0, x1, y1))
        return cropped
    return image

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')
            img = auto_crop(img)
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array.flatten())
    return np.array(images)

def calculate_mean_intensity(images):
    return np.mean(images, axis=1)

mean_intensities = {}

for lang, path in DATASET_DIRS.items():
    imgs = load_images(path)
    mean_intensity = calculate_mean_intensity(imgs)
    mean_intensities[lang] = mean_intensity

plt.figure(figsize=(10, 6))
plt.boxplot(mean_intensities.values(), vert=False, patch_artist=True, labels=mean_intensities.keys())
plt.title('Distribution of Mean Pixel Intensity Across Datasets')
plt.xlabel('Mean Pixel Intensity')
plt.ylabel('Language Classes')
plt.tight_layout()
plt.show()
