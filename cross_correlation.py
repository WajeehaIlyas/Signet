import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def cross_correlation(vec1, vec2):
    vec1 = (vec1 - np.mean(vec1)) / (np.std(vec1) + 1e-8)
    vec2 = (vec2 - np.mean(vec2)) / (np.std(vec2) + 1e-8)
    return np.mean(vec1 * vec2)

avg_vectors = {}
for lang, path in DATASET_DIRS.items():
    imgs = load_images(path)
    avg_vectors[lang] = np.mean(imgs, axis=0)

languages = list(avg_vectors.keys())
n = len(languages)
corr_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        corr_matrix[i, j] = cross_correlation(avg_vectors[languages[i]], avg_vectors[languages[j]])

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Cross-correlation')
plt.xticks(range(n), languages, rotation=45)
plt.yticks(range(n), languages)
plt.title('Cross-Correlation Matrix Between Language Image Sets')
plt.tight_layout()
plt.show()
