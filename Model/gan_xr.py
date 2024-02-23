import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import sys
from Model.base_model import BaseModel
from Model import networks
from Model.networks import *
from Utils.image_pool import ImagePool
from Config.gan_xr_options import gan_xr_v1 
import numpy as np
from Model.losses import *

from Model.gcn import GCN
from Model.segnet import SegNet
from Model.segnet import SegResNet
from Model.deeplabv3_plus import DeepLab
from Model.pspnet import PSPNet
from Model.pspnet import PSPDenseNet
from Model.enet import ENet
from Model.unet import sk,cbam


class GAN_XR_Model(BaseModel):
    def name(self):
        return 'GAN_XR_Model'
    def initialize(self, opt):
        """Initialize the  class.
        Parameters:
            opt (Option class)-- stores all the experiment flags;
        """
        BaseModel.initialize(self, opt)
        #print("-----Initialize Model-----")
        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, \
                                        not opt.no_dropout, opt.init_type, opt.gpu_ids,
                                        n_downsampling=opt.n_downsampling)

        if self.isTrain:
#             self.netD = networks.define_D(opt.input_nc+opt.output_nc, opt.ndf,
#                                             opt.which_model_netD,
#                                             opt.n_layers_D, opt.norm, False, opt.init_type, opt.gpu_ids)

            #self.mask_pool = ImagePool(opt.pool_size) ## Don't need pool anymore
            # define loss functions
            if opt.which_gan_loss == 'gan_loss':
                self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode, tensor=self.Tensor)
            else:
                self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode, tensor=self.Tensor)
            if opt.which_encoder_loss == 'L1':
                self.criterionL1 = torch.nn.L1Loss()   
            elif opt.which_encoder_loss == 'smoothL1':
                self.criterionL1 = torch.nn.SmoothL1Loss()      
            elif opt.which_encoder_loss == 'BCE':
                self.criterionL1 = torch.nn.BCEWithLogitsLoss()
            elif opt.which_encoder_loss == 'CE':
                self.criterionL1 = torch.nn.CrossEntropyLoss()      
            elif opt.which_encoder_loss == '2DCE':
                self.criterionL1 = CrossEntropyLoss2d()  
            elif opt.which_encoder_loss == 'boundaryloss':
                self.criterionL1 = SurfaceLoss()
            elif opt.which_encoder_loss == '2DCE-C':
                self.criterionL1 = CrossEntropyLoss2d()
            elif opt.which_encoder_loss == 'IOU':
                self.criterionL1 = IoULoss()
            elif opt.which_encoder_loss == 'DICE':
                self.criterionL1 = dicelossmulticlass()
            elif opt.which_encoder_loss == 'CEDICE':
                self.criterionL1 = CE_DiceLoss()
            elif opt.which_encoder_loss == 'FocalLoss':
                self.criterionL1 = FocalLoss()
            elif opt.which_encoder_loss == '2DCE-1DIM':
                self.criterionL1 = CrossEntropyLoss2d_1dim()
            elif opt.which_encoder_loss == '2DCE-U':
                self.criterionL1 = CrossEntropyLoss2d_unet()
            elif opt.which_encoder_loss == 'bd-dice':  ## Change loss
                self.criterionS = SurfaceLoss()
                self.criterionD = DiceLoss()
            else:
                self.criterionL1 = torch.nn.L1Loss()               
                
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers = []
            self.schedulers = []
            
            self.optimizers.append(self.optimizer_G)
#             self.optimizers.append(self.optimizer_D)
            
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            if not self.isTrain or opt.continue_train:
                self.load_network(self.netG,'NetG','latest')
#                 self.load_network(self.netD,'NetD','latest')
                
            print('---------- Networks initialized -------------')
            #networks.print_network(self.netG)
            
            #if self.isTrain:
                #networks.print_network(self.netD)
                
            #print('-----------------------------------------------')
    
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        
        self.input_img = input['A' if AtoB else 'B'].to(self.device)
        self.input_mask = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
    def forward(self):
        # G(A)
        
        self.pred_mask = self.netG(self.input_img)
        
    def get_image_paths(self):
        return self.image_paths


    def backward_D(self):
        fake_AB = torch.cat((self.input_img, self.pred_mask), 1)
#         pred_fake = self.netD(fake_AB.detach())
        
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # print(self.input_img.shape,self.input_mask.shape)
        real_AB = torch.cat((self.input_img, self.input_mask), 1)
#         pred_real = self.netD(real_AB)
        
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()



    def backward_G(self):
#         print(self.pred_mask.shape, self.input_mask.shape)
#         fake_AB = torch.cat((self.input_img, self.pred_mask), 1)
#         pred_fake = self.netD(fake_AB)
        
#         self.loss_G_GAN = self.criterionGAN(pred_fake, True)
#         self.loss_G_GAN = 0
    
        # Second, G(A) = B
        if self.opt.which_encoder_loss == 'bd_dice':
            self.loss_G_L1 = (self.criterionS(self.pred_mask, self.input_mask) + self.criterionD(self.pred_mask, self.input_mask)) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = self.criterionL1(self.pred_mask, self.input_mask) * self.opt.lambda_L1
        # combine loss and calculate gradients
#         self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
#         self.set_requires_grad(self.netD, True)  # enable backprop for D
#         self.optimizer_D.zero_grad()     # set D's gradients to zero
#         self.backward_D()                # calculate gradients for D
#         self.optimizer_D.step()          # update D's weights
        # update G
#         self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    
    
    def get_current_visuals(self):
        if self.input_img.shape[0]>1:
            tindex = np.random.randint(0,self.input_img.shape[0])
        else:
            tindex = 0
        
        t_img = (self.input_img[tindex][0].cpu().numpy()/2+0.5)*255
        t_mask = (self.input_mask[tindex][0].cpu().numpy()/2+0.5)*255
        self.netG.eval()
        tout = self.netG(self.input_img[tindex].unsqueeze(0))
        tfout = (tout.data.cpu().numpy()[0][0]/2+0.5)*255
        self.netG.train()
        return {'input_img':t_img,'input_mask':t_mask,'output_mask':tfout}

    
    def get_current_errors(self):
#         ret_errors = OrderedDict([('D', self.loss_D.item()), ('G', self.loss_G.item())])
        ret_errors = OrderedDict([('G', self.loss_G.item())])
        return ret_errors


    def save(self,label):
        self.save_network(self.netG,'NetG',label,self.gpu_ids)
#         self.save_network(self.netD,'NetD',label,self.gpu_ids)
