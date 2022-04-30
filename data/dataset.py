import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import torchvision.transforms.functional as F
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        mask_img = img*(1. - mask) + mask*torch.randn_like(img) #真实图像+掩码盖住的随机部分作为条件图片

        ret['gt_image'] = img
        ret['cond_image'] = mask_img 
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox()) #随机产生掩码
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)



class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        mask_img = img*(1. - mask) + mask*torch.randn_like(img)

        ret['gt_image'] = img
        ret['cond_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

import torchvision.transforms.functional as tf
from data.base_dataset import get_transform

class HarmonizationTrainDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size
        
        self.transform = get_transform()

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        name_parts=path.split('_')
        mask_path = self.imgs[index].replace('composite_images_train','masks') # #修改点5，如果是带噪声的训练，将这里修改为composite_noisy25_images
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.imgs[index].replace('composite_images_train','real_images') # #修改点6，如果是带噪声的训练，将这里修改为composite_noisy25_images
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        comp = self.loader(path)
        real = self.loader(target_path)
        mask = Image.open(mask_path).convert('1')

        comp = tf.resize(comp, [self.image_size[0], self.image_size[1]])
        mask = tf.resize(mask, [self.image_size[0], self.image_size[1]])
        #mask = tf.resize(mask, [224, 224]) #对MAE训练，需要将这里修改为224,224
        real = tf.resize(real,[self.image_size[0],self.image_size[1]])

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        #mask = self.mask_transform(mask)
        mask = tf.to_tensor(mask)

        real = self.transform(real)

        ret['gt_image'] = real
        ret['cond_image'] = comp
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1]

        return ret

    def __len__(self):
        return len(self.imgs)

class HarmonizationTestDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size
        
        self.transform = get_transform()

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        name_parts=path.split('_')
        mask_path = self.imgs[index].replace('composite_images_test','masks') # #修改点5，如果是带噪声的训练，将这里修改为composite_noisy25_images
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.imgs[index].replace('composite_images_test','real_images') # #修改点6，如果是带噪声的训练，将这里修改为composite_noisy25_images
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        comp = self.loader(path)
        real = self.loader(target_path)
        mask = Image.open(mask_path).convert('1')

        comp = tf.resize(comp, [self.image_size[0], self.image_size[1]])
        mask = tf.resize(mask, [self.image_size[0], self.image_size[1]])
        #mask = tf.resize(mask, [224, 224]) #对MAE训练，需要将这里修改为224,224
        real = tf.resize(real,[self.image_size[0],self.image_size[1]])

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        #mask = self.mask_transform(mask)
        mask = tf.to_tensor(mask)

        real = self.transform(real)

        ret['gt_image'] = real
        ret['cond_image'] = comp
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1]

        return ret

    def __len__(self):
        return len(self.imgs)

from lib2to3.pytree import convert
import os.path
import torch
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def pil_loader(path):
    return Image.open(path).convert('RGB')

class HDay2nightTrainDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, data_root)
        self.image_paths = []
        self.isTrain = True
        if self.isTrain==True:
            print('loading training file: ')
            self.trainfile =data_root+'Hday2night_train.txt' #修改点1，替换HCOCO_train.txt
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(data_root, 'composite_noisy25_images', line.rstrip())) #修改点2，增加composite_images，如果是带噪声的训练，将这里修改为composite_noisy25_images
        elif self.isTrain==False:
            print('loading test file')
            self.trainfile = data_root+'Hday2night_test.txt' #修改点3， 替换HCOCO_test
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(data_root, 'composite_noisy25_images', line.rstrip())) #修改点4，如果是带噪声的训练，将这里修改为composite_noisy25_images
        self.transform = get_transform()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        ret = {}

        path = self.image_paths[index]
        name_parts=path.split('_')
        mask_path = self.image_paths[index].replace('composite_noisy25_images','masks') # #修改点5，如果是带噪声的训练，将这里修改为composite_noisy25_images
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.image_paths[index].replace('composite_noisy25_images','real_images') # #修改点6，如果是带噪声的训练，将这里修改为composite_noisy25_images
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        comp = Image.open(path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        comp = tf.resize(comp, [256, 256])
        mask = tf.resize(mask, [256, 256])
        #mask = tf.resize(mask, [224, 224]) #对MAE训练，需要将这里修改为224,224
        real = tf.resize(real,[256,256])

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        #mask = self.mask_transform(mask)
        mask = tf.to_tensor(mask)

        real = self.transform(real)
        #concate the composite and mask as the input of generator
        inputs=torch.cat([comp,mask],0)

        ret['gt_image'] = real
        ret['cond_image'] = comp
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1]

        return ret

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
