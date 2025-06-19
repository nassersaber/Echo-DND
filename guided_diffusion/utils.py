
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# from guided_diffusion.augmentation import rand_augment_transform #, salt_and_pepper_noise

import torch as th
from PIL import Image
import torchvision.utils as vutils

class PadToSquare:
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.functional.pad(img, padding, self.fill, self.padding_mode)

# def get_transform_train(augmentation = False, args=None):
#     tran_list = [PadToSquare(), transforms.Resize((args.image_size,args.image_size))]
#     if augmentation:
#         tran_list.extend([
#             rand_augment_transform(),
#             lambda img: salt_and_pepper_noise(img),
#             transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomHorizontalFlip(),
#         ])
#     tran_list.append(transforms.ToTensor())
#     transform_train = transforms.Compose(tran_list)
#     return transform_train

def get_transform_train(args=None, augmentation=False):
    # Transformations for images only
    resize_tran_list = [PadToSquare(), transforms.Resize((args.image_size,args.image_size))]
    image_tran_list = []
    if augmentation:
        image_tran_list.extend([
            # Gaussian Blur
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1,.2)),
            # Color Jitter
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
        # if random.random()<0.25:
        #     image_tran_list.append(lambda img: salt_and_pepper_noise(img))
    
    # Common transformations for both images and masks
    common_tran_list = []

    # Random flips applied to both images and masks
    # if augmentation:
    #     common_tran_list.extend([
    #         transforms.RandomVerticalFlip(),
    #         transforms.RandomHorizontalFlip(),
    #     ])
    
    common_tran_list.append(transforms.ToTensor())

    transform_resizing = transforms.Compose(resize_tran_list)
    transform_image = transforms.Compose(image_tran_list)
    transform_common = transforms.Compose(common_tran_list)

    return [transform_resizing, transform_image, transform_common]


# def get_transform_for_visualization(augmentation = False, args=None):
#     # tran_list = [PadToSquare(), transforms.Resize((args.image_size,args.image_size))]
#     tran_list = []
#     if augmentation:
#         tran_list.extend([
#             rand_augment_transform(),
#             lambda img: salt_and_pepper_noise(img),
#             transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomHorizontalFlip(),
#         ])
#     tran_list.append(transforms.ToTensor())
#     transform_train = transforms.Compose(tran_list)
#     return transform_train


softmax_helper = lambda x: F.softmax(x, 1)
sigmoid_helper = lambda x: F.sigmoid(x)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def staple(a):
    # a: n,c,h,w detach tensor
    mvres = mv(a)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):
            r = s * mvres
            res = r if i == 0 else torch.cat((res,r),0)
        nres = mv(res)
        gap = torch.mean(torch.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres

def allone(disc,cup):
    disc = np.array(disc) / 255
    cup = np.array(cup) / 255
    res = np.clip(disc * 0.5 + cup,0,1) * 255
    res = 255 - res
    res = Image.fromarray(np.uint8(res))
    return res

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim=True) / b

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

def export(tar, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tar.size(1)
    if c == 3:
        vutils.save_image(tar, fp = img_path)
    else:
        s = th.tensor(tar)[:,-1,:,:].unsqueeze(1)
        s = th.cat((s,s,s),1)
        vutils.save_image(s, fp = img_path)

def norm(t):
    m, s, v = torch.mean(t), torch.std(t), torch.var(t)
    return (t - m) / s
