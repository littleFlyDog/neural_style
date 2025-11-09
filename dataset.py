
from sympy import content, im
import torch
import torchvision
import os
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
#图片读取

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

pretrained_net = torchvision.models.vgg19(weights='IMAGENET1K_V1')
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
pre_model = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])


#返回一张图片在指定层数的特征输出
def extract_features(X, content_layers, style_layers):
    pre_model.to(X.device)
    pre_model.eval()
    contents = []
    styles = []
    for i in range(len(pre_model)):
        X = pre_model[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(content_img):
    contents_Y, _ = extract_features(content_img, content_layers, style_layers)
    return  contents_Y

def get_styles(style_img):
    _, styles_Y = extract_features(style_img, content_layers, style_layers)
    return styles_Y


def load_image(content_path, style_path, image_shape, device):
    content_img = preprocess(Image.open(content_path), image_shape).to(device)
    style_img = preprocess(Image.open(style_path), image_shape).to(device)
    return content_img, style_img


def save_unique_image(image, output_dir = "results", 
                      base_filename= "generated_style", 
                      extension=".png"):
    """
    将 PIL 图片保存到指定目录，文件名后缀数字会自动递增以避免覆盖。

    参数:
    - image (Image.Image): 要保存的 PIL 图片对象。
    - output_dir (str): 保存图片的目录。如果不存在，会自动创建。
    - base_filename (str): 文件名的基础部分。
    - extension (str): 文件扩展名，例如 '.png' 或 '.jpg'。
    """
    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 2. 寻找一个不重复的文件名
    counter = 0
    while True:
        # 构造完整的文件路径
        file_path = os.path.join(output_dir, f"{base_filename}_{counter}{extension}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            # 如果文件不存在，就使用这个路径并跳出循环
            break
        # 如果文件存在，增加计数器，继续下一次循环
        counter += 1

    # 3. 保存图片
    try:
        image.save(file_path)
        print(f"图片成功保存至: {file_path}")
    except Exception as e:
        print(f"保存图片失败: {e}")