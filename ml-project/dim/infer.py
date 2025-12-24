import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import torchvision

import torch
from torch.optim import Adam, SGD
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

import torchvision.transforms as transforms

from model.resnet import resnet18
from mi.losses import local_global_loss
from trainer import Trainer

from utils import save_model, load_model

# parse args
parser = argparse.ArgumentParser()

parser.add_argument('--chkpdir', type=str, default='chkp/')
parser.add_argument('--chkpname', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--outdir', type=str, default='vectors/')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=2)

# load from checkpoint
state = load_model(args.chkpdir, args.chkpname)
model = resnet18()

if device.type == 'cuda':
    print('CUDA device will be used.')
    model = model.cuda()

if args.multi_gpu:
    print('Multi-gpu support is enabled.')
    model = torch.nn.DataParallel(model)
    print ('Using {} gpus.'.format(torch.cuda.device_count()))

model.load_state_dict(state['model'])

vectors = []
#labels = []
for batch in tqdm(loader):
    images, labels_ = batch
    images = images.cuda()

    _, vectors_ = model(images)
    vectors_ = vectors_.cpu().data.numpy()

    #labels.extend(labels_)
    vectors.extend(vectors_)

vectors = np.array(vectors)
torch.save(torch.tensor(vectors), os.path.join(args.outdir, 'vectors.pth'))
