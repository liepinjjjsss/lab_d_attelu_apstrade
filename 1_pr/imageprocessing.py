from PIL import Image
import numpy as np
import math

img_1 = Image.open("source_images/Grey-wolf.jpg").convert("RGB")
img_2 = Image.open("source_images/starry-night-sky-stockcake.jpg").convert("RGB")

pixels_1 = np.array(img_1)
pixels_2 = np.array(img_2)

height_1, width_1, channel_1 = pixels_1.shape
height_2, width_2, channel_2 = pixels_2.shape

result_height = 512
result_width = 512
result_channel = 3

result_pixels = np.zeros((result_height, result_width, result_channel), dtype=np.uint8)

# opacity change

opacity_coefficient = 0.7
for y in range(height_1):
    for x in range(width_1):
        r1, g1, b1 = pixels_1[y, x]
        r2, g2, b2 = pixels_2[y, x]

        r = opacity_coefficient*r1 + (1-opacity_coefficient)*r2
        g = opacity_coefficient*g1 + (1-opacity_coefficient)*g2
        b = opacity_coefficient*b1 + (1-opacity_coefficient)*b2

        result_pixels[y, x] = r, g, b

img = Image.fromarray(result_pixels, mode="RGB")

img.show()
img.save("result_images/opacity_change.jpg")

#differnce 
for y in range(height_1):
    for x in range(width_1):
        r1, g1, b1 = pixels_1[y, x]
        r2, g2, b2 = pixels_2[y, x]

        r = abs(int(r1)-int(r2))
        g = abs(int(g1)-int(g2))
        b = abs(int(b1)-int(b2))

        result_pixels[y, x] = r, g, b

img2 = Image.fromarray(result_pixels, mode="RGB")

img2.show()
img2.save("result_images/difference.jpg")


#soft light
for y in range(height_1):
    for x in range(width_1):
        r1, g1, b1 = pixels_1[y, x]
        r2, g2, b2 = pixels_2[y, x]

        r1_f, r2_f = r1 / 255.0, r2 / 255.0
        g1_f, g2_f = g1 / 255.0, g2 / 255.0
        b1_f, b2_f = b1 / 255.0, b2 / 255.0

        if r1_f <= 0.5:
            r = (2*r1_f-1)*(r2_f-r2_f*r2_f)+r2_f
        else:
            r = (2*r1_f-1)*(math.sqrt(r2_f)-r2_f)+r2_f

        if g1_f <= 0.5:
            g = (2*g1_f-1)*(g2_f-g2_f*g2_f)+g2_f
        else:
            g = (2*g1_f-1)*(math.sqrt(g2_f)-g2_f)+g2_f

        if b1_f <= 0.5:
            b = (2*b1_f-1)*(b2_f-b2_f*b2_f)+b2_f
        else:
            b = (2*b1_f-1)*(math.sqrt(b2_f)-b2_f)+b2_f


        result_pixels[y, x] = r*255, g*255, b*255

img3 = Image.fromarray(result_pixels, mode="RGB")
img3.show()
img3.save("result_images/soft_light.jpg")


#hard light
for y in range(height_1):
    for x in range(width_1):
        r1, g1, b1 = pixels_1[y, x]
        r2, g2, b2 = pixels_2[y, x]

        r1_f, r2_f = r1 / 255.0, r2 / 255.0
        g1_f, g2_f = g1 / 255.0, g2 / 255.0
        b1_f, b2_f = b1 / 255.0, b2 / 255.0

        if r1_f <= 0.5:
            r = 2*r1_f*r2_f
        else:
            r = 1 - 2*(1-r1_f)*(1-r2_f)
        
        if g1_f <= 0.5:
            g = 2*g1_f*g2_f
        else:
            g = 1 - 2*(1-g1_f)*(1-g2_f)

        if b1_f <= 0.5:
            b = 2*b1_f*b2_f
        else:
            b = 1 - 2*(1-b1_f)*(1-b2_f)        

        result_pixels[y, x] = r*255, g*255, b*255

img4 = Image.fromarray(result_pixels, mode="RGB")

img4.show()
img4.save("result_images/hard_light.jpg")



