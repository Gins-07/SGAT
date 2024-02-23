import os
import torch

class gan_xre_unetbase3d():
        dataroot = r'../../Data/train clache patch 512/'
#        dataroot = r'../../Data/train 3500/'
#         dataroot = r'../../Data/800_512_sharp/'
#        dataroot = r'../../Data/DRIVE/'
        checkpoints_dir = '../../Result/lsdriveclacheNT_512_resmvtp_CEDICE/'

        
        
        batchSize = 16
        loadSize = 512# The original value is 286,  The Image size is 
        fineSize = 512
        input_nc = 3
        output_nc = 2
        
        ngf = 32 ##
        ndf = 32
        
        which_encoder_loss = 'CEDICE'
        which_gan_loss = 'gan_loss'
        
        ###Augmentaton parameters
        mean =[0.5,0.5,0.5]       
        std= [0.3,0.3,0.3]
        scale=(0.64, 1)
        ratio_crop=(4. / 5., 5. / 4.)
        ratio_expand=(1,1.6)
        ratio_noise= 30
        
        ###--------------
        
        which_model_netD = 'n_layers' # 'n_layers','piexl'
        n_downsampling = 2  # ADDed By John
        # -------------------
        
        gpu_ids = [0,1]
        name = 'GAN_XR'
        model = 'p2p_xr'
        direction = 'AtoB'
        nThreads = 30

        
        norm = 'instance'
        serial_batches = False
        
#         which_model_netG = 'R2AttU_Net'
#         which_model_netG = 'AttU_Net'
#         which_model_netG = 'PSPNet'
#         which_model_netG = 'sk'
#         which_model_netG = 'GCN'

        # which_model_netG = 'SKTR21'
        which_model_netG = 'resmvtp'

        #************** important change ********************
        dataset_mode = 'aligned_xre'
        lambda_L1 = 1   ## default is 100, trail_1 is 0.001
        #****************************************************

        display_winsize = 256
        display_id = 0       # Do not display 
        display_server = "http://0.0.0.0"
        display_port = 8097
        no_flip = False
        init_type = 'normal'
        no_dropout = False
        phase = 'train_'
        resize_or_crop = 'center_crop'
        isTrain = True
        max_dataset_size = float("inf")
        
#         n_layers_D = 1
        
        gan_mode = 'vanilla' # no_lsgan = False
        display_freq = 100
        display_single_pane_ncols =0
        update_html_freq =1000
        print_freq =100
        save_latest_freq = 5000
        save_epoch_freq = 1  ###   tochange
        
        continue_train = True
        epoch_count = 1
        which_epoch = 'lastest'
        niter = 100
        niter_decay=100
        beta1 = 0.9
#         lr = 0.0001
        lr = 0.0001
        no_html = False
        lr_policy ='lambda'
        lr_decay_iters =50
        n_layers_D = 3
