from fastai_old.core import V
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from unet import Unet, IMAGENET_STD, IMAGENET_MEAN
import glob
import os
from imageio import imwrite

unet = Unet(trained=True, model_name='unet.h5').cuda()
unet.eval()

normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
toTensor = transforms.ToTensor()
tfm = transforms.Compose([
    transforms.Resize(256),
    toTensor,
    normalize
])


def generate_segs(image_folder):
    images = glob.glob(os.path.join(image_folder, '*.png'))
    kernel = np.ones((10, 10))
    os.makedirs(f'{image_folder.rstrip("/")}_cropped/', exist_ok=True)
    for image_file in tqdm(images):
        image = Image.open(image_file).convert('RGB')

        image_v = V(tfm(image)[None])
        py = torch.sigmoid(unet(image_v))
        py = (py[0].cpu() > 0.5).type(torch.FloatTensor)
        mask = py[0].numpy()
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(np.bool)

        image = cv2.resize(np.asarray(image), (256, 256))
        mask = np.dstack([mask, mask, mask])
        imwrite(f'{image_folder.rstrip("/")}_cropped/{os.path.basename(image_file)}', image * mask)
