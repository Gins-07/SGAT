import os.path
import random
import torchvision.transforms as transforms
import torch
from Data.base_dataset import BaseDataset, get_transform
from Data.image_folder import make_dataset
from PIL import Image


class XRDataset_Seq(BaseDataset):
    """
    Dataset Format:
        000001_20.jpg  the annotated frame
        A is Smaple, B is Mask
    """
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.fr_int = opt.fr_int
        self.fr_param = (opt.input_nc-1)//2
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.A_size = len(self.B_paths)
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.transform = get_transform(opt)
        
        
    def __getitem__(self, index):
        tindex = index % self.A_size +1
        A_paths = [self.dir_A+'/%06d_%02d.jpg'%(tindex,tindex2) for tindex2 in range(20-self.fr_int*self.fr_param,20+self.fr_int*self.fr_param+1,self.fr_int)]
        B_path = self.dir_B + '/%06d.jpg'%(tindex)
        #A_path = self.A_paths[index % self.A_size]
        #B_path = self.B_paths[index % self.A_size]
        As= [Image.open(t_A_path) for t_A_path in A_paths]
        
        if torch.randint(0,10,[1])[0]>5:
            As.reverse()
        
        B = Image.open(B_path)
        if torch.randint(0,10,[1])[0]>5:
            As = [A.transpose(Image.FLIP_LEFT_RIGHT) for A in As]
            B = B.transpose(Image.FLIP_LEFT_RIGHT)
        if torch.randint(0,10,[1])[0]>5:
            As = [A.transpose(Image.FLIP_TOP_BOTTOM) for A in As]
            B = B.transpose(Image.FLIP_TOP_BOTTOM)
        As = [A.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC) for A in As]
        B = B.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        As = [transforms.ToTensor()(A) for A in As]
        B = transforms.ToTensor()(B)
        As = torch.cat([transforms.Normalize((0.5,), (0.5,))(A) for A in As],0)
        B = transforms.Normalize((0.5,), (0.5,))(B)
        
        return {'A': As, 'B': B,
                'A_paths': A_paths, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_XR_Seq'
