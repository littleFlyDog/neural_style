from dataset import get_contents, get_styles,load_image 
from train import train_fn 
from model import SynthesizedImage, gram
import torch
from dataset import postprocess, save_unique_image
lr=0.3
lr_decay_epoch=50
num_epochs=700

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
content_path, style_path = 'test1.jpg', 'autumn-oak.jpg'
image_shape = (300, 450)

content_img, style_img = load_image(content_path, style_path, image_shape, device)#这里是已经处理好的张量

model = SynthesizedImage(content_img.shape).to(device)
model.weight.data.copy_(content_img.data)

trainer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)

contents_X = get_contents(content_img)
styles_X = get_styles(style_img)
styles_X_gram = [gram(x) for x in styles_X]
output = train_fn(model, contents_X, styles_X_gram, num_epochs, trainer, scheduler)

final_img = postprocess(output.cpu().detach())
final_img.show()
save_unique_image(final_img)