from model import *
import torch
import cv2
from matplotlib import pyplot as plt
import torchvision
import torch.nn.functional as F
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = ResNet18(Residual).to(device)

model_path = 'E:\PYTHONfile\PYTorch\\video\models\\best_resnet.pth'

resnet.load_state_dict(torch.load(model_path, map_location=device))

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
resnet.eval()
# with torch.no_grad():
#     img = cv2.imread('11.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     print(img.shape)
#
#     # 显示图片
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
#     img_tensor = torch.Tensor(img)
#     img_tensor = img_tensor.to(device)
#     img_tensor = img_tensor.permute(2, 0, 1)
#     img_tensor = img_tensor.unsqueeze(0)
#     output = resnet(img_tensor)
#     pre = output.argmax(1)
#     print(classes[pre[0]])
#     print('ok')

for data in test_data:
    img, label = data
    img_plt = img.permute(1, 2, 0)
    plt.imshow(img_plt)
    plt.axis('off')
    plt.show()
    img = img.to(device)
    img = img.unsqueeze(0)
    output = resnet(img)
    pred = output.argmax(dim=1, keepdim=True)
    print('预测类别是：{}'.format(classes[pred[0]]))
    print('真实类别是：{}'.format(classes[label]))
    time.sleep(5)