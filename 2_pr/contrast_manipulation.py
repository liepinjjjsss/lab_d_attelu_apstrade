from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import os

def show_histogram(pixels, filename, method, beforeafter):
    R = pixels[:, :, 0]
    G = pixels[:, :, 1]
    B = pixels[:, :, 2]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(pixels)
    plt.title(f"{filename} {beforeafter} {method} correction")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    colors = ["r", "g", "b"]

    for i, c in enumerate(colors):
        plt.hist(
            pixels[:, :, i].flatten(),
            bins=256,
            range=(0, 255),
            color=c,
            alpha=0.5
        )

    plt.title(f"RGB histogram - {filename} {beforeafter} {method} correction")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")

    plt.tight_layout()
    plt.savefig(f"2_pr/histograms/{filename}_{method}_{beforeafter}.png")
    plt.show()

#gamma correction
folder = "2_pr/source_images"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)

    if os.path.isfile(file_path):
        image = Image.open(file_path).convert("RGB")

        pixels = np.array(image)
        result_pixels = np.zeros_like(pixels)

        height, width, channel = pixels.shape

        show_histogram(pixels, os.path.splitext(filename)[0], "gamma", "before")
        gamma = 0.7
        for y in range(height):
            for x in range(width):
                r, g ,b = pixels[y, x]

                intensity_avg = (int(r)+int(g)+int(b))/3
                new_intensity = 255 * (intensity_avg/255)**gamma

                if intensity_avg > 0:
                    scale = new_intensity/intensity_avg
                else:
                    scale = 0

                r = r*scale
                g = g*scale
                b = b*scale

                result_pixels[y,x] = r,g,b
        show_histogram(result_pixels, os.path.splitext(filename)[0], "gamma", "after")

#histogram equalization

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)

    if os.path.isfile(file_path):
        image = Image.open(file_path).convert("RGB")
        pixels = np.array(image)
        rgb_pixels = np.array(image).astype(np.float32)
        hsv_pixels = np.zeros(shape=rgb_pixels.shape, dtype=np.float32)

        show_histogram(pixels, os.path.splitext(filename)[0], "HE", "before")

        height, width, channel = rgb_pixels.shape
        for y in range(height):
            for x in range(width):
                r, g, b = rgb_pixels[y, x]/255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv_pixels[y, x] = h, s, v

        V = hsv_pixels[:, :, 2]
        V_uint8 = (V*255).astype(np.uint8)

        hist = np.zeros(256)

        for value in V_uint8.flatten():
            hist[value] += 1

        pdf = hist/hist.sum()
        cdf = np.cumsum(pdf)

        V_eq = np.zeros_like(V_uint8)

        for y in range(height):
            for x in range(width):
                old_V = V_uint8[y, x]
                V_eq[y,x] = int(255* cdf[old_V])

        hsv_pixels[:,:, 2] = V_eq/255
        rgb_out = np.zeros_like(hsv_pixels)

        for y in range(height):
            for x in range(width):
                h, s, v = hsv_pixels[y, x]
                r, g ,b = colorsys.hsv_to_rgb(h, s, v)
                rgb_out[y, x] = r, g ,b

        rgb_out = (rgb_out * 255).clip(0, 255).astype(np.uint8)

        show_histogram(rgb_out, os.path.splitext(filename)[0], "HE", "after")


