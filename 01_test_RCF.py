import os, sys, glob, pathlib
import numpy as np
import os.path as osp
import cv2
import argparse
import torch
import datetime
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader
import torchvision

sys.path.append('../packages/RCF_pytorch_author')
# from dataset import MuscleFiber_Dataset
from models import RCF

def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im


class MuscleFiber_Dataset(torch.utils.data.Dataset):
    """
    Test Dataset Muscle Fibre
    """
    def __init__(self, dataset='', transform=False):
        self.transform = transform
        self.filelist = dataset
        self.divider = 2
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        # remove leading white spaces 
        img_file = self.filelist[index].rstrip()
        img = np.array(Image.open(img_file).convert('RGB'), dtype=np.float32)
        img = resize(img, (img.shape[0] // self.divider, img.shape[1] // self.divider),
                       anti_aliasing=True)
        img = prepare_image_PIL(img)
        return img


def single_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        all_res = torch.zeros((len(results), 1, H, W))
        
        for i in range(len(results)):
            all_res[i, 0, :, :] = results[i]
        filename = osp.splitext(test_list[idx])[0]
        image_dir = os.path.basename(os.path.dirname(test_list[idx]))
        pathlib.Path(os.path.join(save_dir, image_dir)).mkdir(parents=True, exist_ok=True)
        
        torchvision.utils.save_image(1 - all_res, osp.join(save_dir, image_dir, '%s.jpg' % osp.basename(filename)))
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = (fuse_res * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, image_dir, '%s.png' % osp.basename(filename)), fuse_res)
        #print('\rRunning single-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running single-scale test done')

def multi_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
        
    scale = [0.5, 1, 1.5]
    for idx, image in enumerate(test_loader):
        in_ = image[0].numpy().transpose((1, 2, 0))
        _, _, H, W = image.shape
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
        ms_fuse = ms_fuse / len(scale)
        ### rescale trick
        # ms_fuse = (ms_fuse - ms_fuse.min()) / (ms_fuse.max() - ms_fuse.min())
        filename = osp.splitext(test_list[idx])[0]
        image_dir = os.path.basename(os.path.dirname(test_list[idx]))
        pathlib.Path(os.path.join(save_dir, image_dir)).mkdir(parents=True, exist_ok=True)
        
        result_out = ((1 - ms_fuse) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, image_dir, '%s_ms.jpg' % osp.basename(filename)), result_out)
        result_out2 = (ms_fuse * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, image_dir, '%s_ms.png' % osp.basename(filename)), result_out2)
        #print('\rRunning multi-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running multi-scale test done')

    
class Args:
    gpu = '1'
    checkpoint = '../packages/RCF_pytorch_author/models/bsds500_pascal_model.pth'
    save_dir = '../Results'
    dataset='../data/Images'

args = Args()
date = str(datetime.date.today())
save_dir = osp.join(args.save_dir, date)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if not osp.isdir(args.save_dir):
    os.makedirs(args.save_dir)
    
image_datasets = glob.glob(osp.join(args.dataset, '*','*.jpg'),recursive=True)

test_dataset  = MuscleFiber_Dataset(dataset=image_datasets)
test_loader   = DataLoader(test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False)

model = RCF().cuda()

if osp.isfile(args.checkpoint):
    print("=> loading checkpoint from '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    print("=> checkpoint loaded")
else:
    print("=> no checkpoint found at '{}'".format(args.checkpoint))

print('Performing the testing...')
single_scale_test(model, test_loader, image_datasets, os.path.join(save_dir, 'test'))
multi_scale_test(model, test_loader, image_datasets, os.path.join(save_dir, 'multitest'))