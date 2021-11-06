# coding=utf-8
from __future__ import absolute_import, division, print_function
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torch.utils.data as data
from PIL import Image, ImageFile
from torch.optim import optimizer
from pathlib import Path
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
from functions import GramMSELoss

from sampler import InfiniteSamplerWrapper
import models.modeling as modeling
from functions import calc_content_loss, calc_style_loss, train_transform, normal, inverse_normalize
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
# Required parameters
parser.add_argument('--content', type=str,#default='input/content/golden_gate.jpg',
                    help='File path to the content image')
parser.add_argument('--style', type=str, #default='input/style/la_muse.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')

# Additional options
parser.add_argument("--lr", default=1e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--lr_decay", default=1e-5, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument('--size', type=int, default=512,
                    help='New size for the content and style images, \
                    keeping the original size if set to 0')
parser.add_argument("--num_steps", default=4000, type=int,
                    help="Total number of training iterations to perform.")
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
parser.add_argument("--ics_init", type=str, default='content',
                    help="the initial Ics. Can be: content, random. default as content.")  
parser.add_argument("--encoder_dir", type=str, default=None,
                    help="Where to search for trained encoder checkpoint.")
parser.add_argument("--style_compute", type=str, default='adain',
                    help="style transfer method. adain or gram. default as adain.")
parser.add_argument("--eval_every", default=500, type=int,
                    help="Save training samples every so many steps."
                    "Will always run one evaluation at the end of training.")
parser.add_argument("--content_weight", default=1, type=float,
                    help="The weight of content loss.")
parser.add_argument("--style_weight", default=100, type=float,
                    help="The weight of style loss.")
parser.add_argument("--debug", action='store_true', default=False,
                    help="start with debug mode, without any output, log and checkpoints.")
args = parser.parse_args()
args.log_dir = './logs/'+localtime
# Setup CUDA, GPU & distributed training

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.image_save_dir = './nst_images/'+localtime
if args.debug == False:
    writer = SummaryWriter(log_dir=args.log_dir)

    if not os.path.exists(args.image_save_dir):
        os.makedirs(args.image_save_dir)

assert (args.content and args.style), 'Both content and style image directory should be provided!'
# #my_Vit  ViT-B_16_h ViT-B_16
config = modeling.CONFIGS[args.model]
model = modeling.VisionTransformer(config,pe_mode=args.pe_mode)
assert (args.encoder_dir), 'The encoder checkpoint should be provided!'
model.transformer.load_state_dict(torch.load(args.encoder_dir))


print("Model created!")
model.to(args.device)
#model = torch.nn.DataParallel(model, device_ids=[0,1])
mseloss = nn.MSELoss()
smoothl1loss = nn.SmoothL1Loss()
for param in model.transformer.parameters():
    param.requires_grad = False


vgg = modeling.vgg
vgg.load_state_dict(torch.load('./models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:44])
with torch.no_grad():
    loss_net = modeling.Net(vgg)
loss_net.to(args.device)
gramloss = GramMSELoss()

style_image = Image.open(args.style).convert('RGB')
content_image = Image.open(args.content).convert('RGB')
content_tf = test_transform(512, True)
style_tf = test_transform(512, True)

content_image = content_tf(content_image).unsqueeze(0)
style_image = style_tf(style_image).unsqueeze(0)
if args.ics_init=='content':
    Ics = Variable(content_image.data.clone())
elif args.ics_init=='random':
    Ics = Variable(torch.rand_like(content_image))



if not os.path.exists(args.image_save_dir):
    os.makedirs(args.image_save_dir)
output_image = torch.cat((content_image,style_image,Ics))
save_image(output_image, args.image_save_dir + "/iter_"  + str(0) + "_th image.jpg")

content_image = content_image.to(device)
style_image = style_image.to(device)
Ics = Ics.to(device)
Ics.requires_grad = True
optimizer = torch.optim.Adam([Ics], lr = args.lr)

style_feats,_ = model.transformer.nst(style_image)
content_feats,_ = model.transformer.nst(content_image)

from datetime import datetime
time0 = datetime.now()
for i in tqdm(range(args.num_steps)):

    Ics_feats,_ = model.transformer.nst(Ics)
    loss_c = calc_content_loss(content_feats[3], Ics_feats[3]) + calc_content_loss(content_feats[2], Ics_feats[2])
    if args.style_compute=='gram':
        print('gram')
        loss_s = gramloss(Ics_feats[-1], style_feats[-1])
        for j in range(1, 4):
            loss_s += gramloss(Ics_feats[0-j], style_feats[0-j])
    else:
        print('adain')
        loss_s = calc_style_loss(Ics_feats[-1], style_feats[-1])
        for j in range(1, 4):
            loss_s += calc_style_loss(Ics_feats[0-j], style_feats[0-j])

    loss = args.style_weight * loss_s + args.content_weight * loss_c
    print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
        )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if args.debug == False:
        writer.add_scalar('loss_c',loss_c.item(),i+1)
        writer.add_scalar('loss_s',loss_s.item(),i+1)
        writer.add_scalar('loss',loss.item(),i+1)

    if ((i+1) % args.eval_every == 0 or (i+1)== args.num_steps) and (args.debug == False):
        output_image = torch.cat((content_image,style_image,Ics))
        save_image(output_image, args.image_save_dir + "/iter_"  + str(i+1) + "_th image.jpg")
time1 = datetime.now()
print((time1-time0))
if args.debug == False:
    writer.close()
