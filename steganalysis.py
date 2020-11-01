import torch
from model import Hide, Reveal
# from utils import DatasetFromFolder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
# import pytorch_msssim
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import math
#Crop the picture and convert it to tensor form
def transforms_img(img):
    img = Image.open(img) #read image
    img = img.resize((256, 256))
    tensor = transforms.ToTensor()(img) #Convert image to tensor，
    # print(tensor.shape) # [3, 224, 224]
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
    # print(tensor.shape)# [1,3, 224, 224]
    # tensor = tensor.cuda()
    return tensor
# Import the model
def moxing(reveal):
    reveal_net = Reveal()
    reveal_net.eval()
    # reveal_net.cuda()

    reveal_net.load_state_dict(torch.load(reveal))
    return reveal_net
#Model indicator loss function calculation
def norm(img1, img2):
    criterion = nn.MSELoss()
    loss_r = criterion(img1,img2)
    print("loss_r:" + str(loss_r.item()))
    SSIM_h=1-pytorch_msssim.ssim(img1,img2)
    print("SSIM_h:"+str(SSIM_h.item()))
    MS_SSIM_h=1-pytorch_msssim.ms_ssim(img1,img2)
    print("MS_SSIM_h:"+str(MS_SSIM_h.item()))
def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100

    return 10 * math.log10(255.0 ** 2 / mse)

if __name__ == '__main__':
    # MSE Loss SSIM MS_SSIM
    reveal='./checkpoint/epoch_290_reveal.pkl'
    reveal_net=moxing(reveal)
    
    output='./result1/output.png'
    output=transforms_img(output)

    reveal_secret = reveal_net(output)

    # secret = './result1/secret.png'
    # secret = transforms_img(secret)
    # norm(secret, reveal_secret)

    save_image(reveal_secret.cpu().data[:4],fp='./result1/reveal_secret11.png')
    # gt = cv2.imread('./result1/cover.png')
    # img = cv2.imread('./result1/output.png')
    # print(psnr1(gt, img))

