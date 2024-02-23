import os.path
import random
import torchvision.transforms as transforms
import torch
from Data.base_dataset import BaseDataset, get_transform
from Data.image_folder import make_dataset
from PIL import Image
from Data.augmentation import *


class WnetAugmentation(object):
    def __init__(self, size=256, mean=(0.5),std=(0.3),\
                 scale=(0.64, 1),\
                 ratio_crop=(4. / 5., 5. / 4.),\
                 ratio_expand=(1,2),\
                 ratio_noise=30):
        self.augment = Compose([
                                ConvertFromInts(),
                                Expand(0,ratio = ratio_expand), # expand ratio
                                PhotometricDistort_grey(delta=ratio_noise,),   ## delta control noise intensity
                                RandomResizedCrop(size=size,scale=scale,ratio=ratio_crop), ## scale
                                Resize(size),
                                RandomMirror(),
                                ToTensor(),
                                Normalize(mean,std),
                            ])

    def __call__(self, img, boxes, labels, masks):
        return self.augment(img, boxes, labels, masks)
    
    
class XRDataset(BaseDataset):
    """
    Enhanced version of XRDataset:
    with Dataaugmentation
    """
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.A_size = len(self.A_paths)
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.transform = self.get_transform()
        
        assert(len(self.A_paths)==len(self.B_paths))
    
    def get_transform(self):
        return WnetAugmentation(size=self.opt.fineSize,
                                mean=self.opt.mean,std=self.opt.std,\
                                 scale=self.opt.scale,\
                                 ratio_crop=self.opt.ratio_crop,\
                                 ratio_expand=self.opt.ratio_expand,\
                                 ratio_noise=self.opt.ratio_noise)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        A= Image.open(A_path)
        B = Image.open(B_path)
        #if torch.randint(0,10,[1])[0]>5:
            #A = A.transpose(Image.FLIP_LEFT_RIGHT)
            #B = B.transpose(Image.FLIP_LEFT_RIGHT)
        #A = A.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        #B = B.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        #A = transforms.ToTensor()(A)
        #B = transforms.ToTensor()(B)
        #A = transforms.Normalize((0.5,), (0.5,))(A)
        #B = transforms.Normalize((0.5,), (0.5,))(B)
        At,_,_,Bt = self.transform(A,np.array([[0,0,1,1]]),np.array([[4]]),B) 

        return {'A': At, 'B': Bt.unsqueeze(0),'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_XR'
