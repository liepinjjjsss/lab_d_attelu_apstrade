from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def salt_pepper_noise(filepath, noise_level):
    image = Image.open(filepath)
    pixels = np.array(image)
    result_pixels = np.zeros_like(image)

    height, width, channels = pixels.shape

    for y in range(height):
        for x in range(width):
            random_number = random.random()
            if random_number > 1-(noise_level/2):
                r = 0
                g = 0
                b = 0
                result_pixels[y, x] = r, g, b
            elif random_number < (noise_level/2):
                r = 255
                g = 255
                b = 255
                result_pixels[y, x] = r, g, b
            else:
                result_pixels[y, x] = pixels[y, x]
    
    # plt.figure(figsize=(5, 5))
    # plt.axis("off")
    # plt.imshow(result_pixels)
    # plt.show()

    image = Image.fromarray(result_pixels)
    image.show()
    image.save(os.path.join("3_pr/result_images", f"salt_pepper_{os.path.basename(filepath)}"))


def uniform_noise(filepath, noise_level):
    image = Image.open(filepath)
    pixels = np.array(image)
    result_pixels = np.zeros_like(image)

    noise = np.random.randint(-noise_level, noise_level, size = pixels.shape, dtype = np.int16)
    result_pixels = pixels.astype(np.int16) + noise

    result_pixels = np.clip(result_pixels, 0, 255).astype(np.uint8)
    
    # plt.figure(figsize=(5, 5))
    # plt.axis("off")
    # plt.imshow(result_pixels)
    # plt.show()

    image = Image.fromarray(result_pixels)
    image.show()
    image.save(os.path.join("3_pr/result_images", f"uniform_{os.path.basename(filepath)}"))


def mean_filter_denoise(filepath):
    image = Image.open(filepath)
    pixels = np.array(image)
    result_pixels = np.zeros_like(pixels)
    height, width, channel = pixels.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            r = np.int32(0)
            g = np.int32(0)
            b = np.int32(0)
            for j in range(y - 1, y + 2):
                for k in range(x - 1, x + 2):
                    r += pixels[j, k, 0]
                    g += pixels[j, k, 1]
                    b += pixels[j, k, 2]
            result_pixels[y, x] = r // 9, g // 9, b // 9 

    # return Image.fromarray(result_pixels.astype(np.uint8))
    # plt.figure(figsize=(5, 5))
    # plt.axis("off")
    # plt.imshow(result_pixels)
    # plt.show()
        
    image = Image.fromarray(result_pixels)
    image.show()
    image.save(os.path.join("3_pr/denoised_images",f"mean_denoise_{os.path.basename(filepath)}" ))

def median_filter_denoise(filepath):
    image = Image.open(filepath)
    pixels = np.array(image, dtype=np.int32)
    result_pixels = np.zeros_like(pixels)
    height, width, channel = pixels.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            r_vals = []
            g_vals = []
            b_vals = []
            for j in range(y - 1, y + 2):
                for k in range(x - 1, x + 2):
                    r_vals.append(pixels[j, k, 0])
                    g_vals.append(pixels[j, k, 1])
                    b_vals.append(pixels[j, k, 2])
            result_pixels[y, x, 0] = sorted(r_vals)[4]  # middle of 9 values
            result_pixels[y, x, 1] = sorted(g_vals)[4]
            result_pixels[y, x, 2] = sorted(b_vals)[4]

    image = Image.fromarray(result_pixels.astype(np.uint8))
    image.show()
    image.save(os.path.join("3_pr/denoised_images",f"median_denoise_{os.path.basename(filepath)}" ))

KERNEL = [
    [1,  2,  1],
    [2,  4,  2],
    [1,  2,  1]
]
KERNEL_SUM = 16  # sum of all kernel values

def gaussian_filter_denoise(filepath):
    image = Image.open(filepath)
    pixels = np.array(image, dtype=np.int32)
    result_pixels = np.zeros_like(pixels)
    height, width, channel = pixels.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            r = np.int32(0)
            g = np.int32(0)
            b = np.int32(0)
            for j in range(y - 1, y + 2):
                for k in range(x - 1, x + 2):
                    weight = KERNEL[j - (y - 1)][k - (x - 1)]
                    r += pixels[j, k, 0] * weight
                    g += pixels[j, k, 1] * weight
                    b += pixels[j, k, 2] * weight
            result_pixels[y, x] = r // KERNEL_SUM, g // KERNEL_SUM, b // KERNEL_SUM

    image = Image.fromarray(result_pixels.astype(np.uint8))
    image.show()
    image.save(os.path.join("3_pr/denoised_images",f"gaussian_denoise_{os.path.basename(filepath)}" ))



folder = "3_pr/source_images"
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)

    salt_pepper_noise(filepath, 0.05)
    uniform_noise(filepath, 40)

folder = "3_pr/result_images"
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)

    mean_filter_denoise(filepath)
    median_filter_denoise(filepath)
    gaussian_filter_denoise(filepath)





