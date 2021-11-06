import argparse
import os
import random
import time
from os.path import basename, splitext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import center_crop
from torchvision.transforms.transforms import CenterCrop, RandomCrop
from torchvision.utils import save_image

import models.modeling as modeling
from functions import inverse_normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,# default='input/content/golden_gate.jpg',
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,# default='data/content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, #default='input/style/la_muse.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,# default='data/style',
                    help='Directory path to a batch of style images')

# Additional options
parser.add_argument('--size', type=int, default=512,
                    help='New size for the content and style images, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='test_image/'+localtime,
                    help='Directory to save the output image(s)')
parser.add_argument("--model", type=str, default='ViT-B_16',
                    help="The ViT model type.")
parser.add_argument("--pe_mode", type=str, default=None,
                    help="The position embedding. Can be: learnable, sin. default as None.")    
parser.add_argument("--upsampling_mode", type=str, default=None,
                    help="Set upsampling way. can be mlp. default as None for cnns.") 
parser.add_argument("--encoder_dir", type=str, default=None,
                    help="Where to search for trained encoder checkpoint.")
parser.add_argument("--decoder_dir", type=str, default=None,
                    help="Where to search for trained decoder checkpoint.")
parser.add_argument("--style_transfer", type=str, default='adain',
                    help="style transfer method. adain or wct.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content or args.content_dir)
assert (args.style or args.style_dir)
if not args.content:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]
else:
    content_paths = [args.content]
if not args.style:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]
else:
    style_paths = [args.style]


model = modeling.VisionTransformer(modeling.CONFIGS[args.model],pe_mode=args.pe_mode,upsample=args.upsampling_mode)
assert (os.path.isfile(args.encoder_dir)), 'The valid pretrained ViT encoder checkpoint should be provided!'
assert (os.path.isfile(args.decoder_dir)), 'The valid pretrained ViT decoder checkpoint should be provided!'
model.transformer.load_state_dict(torch.load(args.encoder_dir),strict=False)
model.transformerdecoder.load_state_dict(torch.load(args.decoder_dir),strict=False)
print("Load checkpoints.")

model.to(device)
model.eval()
print("load done!")
from math import ceil

crop='store_true'
content_tf = test_transform(args.size, crop)
style_tf = test_transform(args.size, crop)

from datetime import datetime

time0 = datetime.now()
count = 0
for content_path in content_paths:
    for style_path in style_paths:
        
        content = content_tf(Image.open(content_path).convert("RGB"))

        h,w,c=np.shape(content)    
        style = style_tf(Image.open(style_path).convert("RGB"))

      
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        
        with torch.no_grad():
            output= model(content,style,args.style_transfer)
        
        output = torch.cat((content.cpu(), style.cpu(), output.cpu()))
        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            args.output, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0], args.save_ext
        )
        content_name = '{:s}/{:s}{:s}'.format(
            args.output, splitext(basename(content_path))[0],
            args.save_ext
        )
        style_name = '{:s}/{:s}{:s}'.format(
            args.output, splitext(basename(style_path))[0],
            args.save_ext
        )

 
        save_image(output, output_name)
        print(output_name)
        count+=1
time1 = datetime.now()
print(time1-time0)
print(count)

