import re
import matplotlib.pyplot as plt

def parse_loss_file(file_path):
    # 初始化用于存储生成器和判别器loss的列表
    epochs = []
    loss_D = []
    loss_G = []
    
    # 打开文件读取每一行
    with open(file_path, 'r') as f:
        for line in f:
            # 使用正则表达式匹配 epoch, 判别器loss和生成器loss
            match = re.match(r'\[(\d+)/\d+\] total_Loss_D: ([\d.]+) total_Loss_G ([\d.]+)', line)
            if match:
                # 提取epoch，判别器loss和生成器loss
                epoch = int(match.group(1))
                d_loss = float(match.group(2))
                g_loss = float(match.group(3))
                
                # 将这些值添加到对应的列表中
                epochs.append(epoch)
                loss_D.append(d_loss)
                loss_G.append(g_loss)
    
    return epochs, loss_D, loss_G

def plot_losses(epochs, loss_D, loss_G):
    # 绘制判别器和生成器的loss曲线
    plt.figure(figsize=(10, 5))
    
    # 判别器loss曲线
    plt.plot(epochs, loss_D, label="Discriminator Loss", color='red', linestyle='--')
    
    # 生成器loss曲线
    plt.plot(epochs, loss_G, label="Generator Loss", color='blue', linestyle='-')
    
    # 添加标题和标签
    plt.title("Discriminator and Generator Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # 添加图例
    plt.legend()
    
    # 显示图像
    plt.grid(True)
    # plt.show()
    plt.savefig('GANLoss.png')

# 指定损失文件的路径
loss_file = './Models/GAN/log.txt'

# 解析损失文件
epochs, loss_D, loss_G = parse_loss_file(loss_file)

# 绘制损失曲线
plot_losses(epochs, loss_D, loss_G)