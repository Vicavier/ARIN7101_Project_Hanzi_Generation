from PIL import Image
import os

def split_image(image_path, output_folder, nrow=7, img_size=32, padding=2):
    """
    将大图拆分为多个小图并保存
    参数:
        image_path (str): 大图的路径
        output_folder (str): 保存小图的文件夹路径
        nrow (int): 每行的图片数量（默认 7）
        img_size (int): 小图的尺寸（默认 32x32）
        padding (int): 图片之间的间距（默认 2）
    """
    # 打开大图
    img = Image.open(image_path)
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base = 147
    # 遍历每一张小图，按照行和列的顺序拆分
    for i in range(nrow):
        for j in range(nrow):
            # 计算小图在大图中的位置，去掉 padding
            left = j * (img_size + padding) + padding
            upper = i * (img_size + padding) + padding
            right = left + img_size
            lower = upper + img_size
            
            # 裁剪出小图
            cropped_img = img.crop((left, upper, right, lower))
            
            # 保存小图，命名方式为 "image_00.png", "image_01.png", ...
            cropped_img.save(os.path.join(output_folder, f"长_{base}.png"))
            base += 1

# 示例使用
image_path = './Models/DDPM/img/长4.png'  # 大图的路径
output_folder = './Models/DDPM/split_img'  # 保存小图的文件夹
split_image(image_path, output_folder)