import os

# 指定文件夹路径
folder_path = './Models/GAN/he_test'

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 遍历文件并进行重命名
for index, filename in enumerate(files):
    # 构造新的文件名
    new_name = f"禾{index + 1}.png"
    
    # 获取旧文件的完整路径和新文件的完整路径
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)
    
    # 重命名文件
    os.rename(old_file, new_file)

print("文件重命名完成！")