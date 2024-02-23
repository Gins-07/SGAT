import os.path
import random
import torchvision.transforms as transforms
import torch
from Data.base_dataset import BaseDataset, get_transform
from Data.image_folder import make_dataset
from PIL import Image
import random

class XRDataset(BaseDataset):
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
        self.transform = get_transform(opt)
        
        assert(len(self.A_paths)==len(self.B_paths))
        
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        A= Image.open(A_path)
        B = Image.open(B_path)
        A = A.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        B = B.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        
#         deg = random.randint(0,360)
#         A = transforms.RandomRotation((deg,deg))(A)
#         B = transforms.RandomRotation((deg,deg))(B)
        
        if torch.randint(0,10,[1])[0]>5:
            A = A.transpose(Image.FLIP_LEFT_RIGHT)
            B = B.transpose(Image.FLIP_LEFT_RIGHT)
        if torch.randint(0,10,[1])[0]>5:
            A = A.transpose(Image.FLIP_TOP_BOTTOM)
            B = B.transpose(Image.FLIP_TOP_BOTTOM)   
    
    
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        
        A = transforms.Normalize((0.5,), (0.5,))(A)
        B = transforms.Normalize((0.5,), (0.5,))(B)
        
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_XR'
