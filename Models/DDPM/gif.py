from PIL import Image
import os
import re

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]
images = []
png_files = [f for f in os.listdir('./Models/DDPM/sampling_progress') if f.endswith(".png")]
for filename in sorted(png_files, key=natural_sort_key,reverse=True):
    if filename.endswith(".png"):
        file_path = os.path.join('./Models/DDPM/sampling_progress', filename)
        img = Image.open(file_path)
        images.append(img)

images[0].save("./Models/DDPM/sampling_process.gif", save_all=True, append_images=images[1:], duration=150, loop=False)