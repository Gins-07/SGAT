import os
import torch

class gan_xrs_v1():
        """
        For sequence 
            new parameters :  frame_interval 
        """
        #dataroot = os.path.expanduser('~')+'/Code/XR_project/Data/20frame/'
        dataroot = '../../../../XR_project/Data/20frame480/'
        batchSize = 32
        loadSize = 256 # The original value is 286,  The Image size is 
        fineSize = 256
        input_nc = 3
        output_nc = 1
        
        ngf = 32
        ndf = 32
        
        fr_int = 1

        which_model_netD = 'n_layers' # 'n_layers','piexl'
        n_downsampling = 2  # ADDed By John

        gpu_ids = [0,1]
        name = 'GAN_XR_with_polar_transform'
        model = 'p2p_xr'
        direction = 'AtoB'
        nThreads = 4
        #checkpoints_dir = os.path.expanduser('~')+'/Code/XR_project/Result/Trial_seq_0/'
        checkpoints_dir = '../../Result/3f_100_9'
        norm = 'instance'
        serial_batches = False
        which_model_netG = 'resnet_9blocks'

        #************** important change ********************
        dataset_mode = 'aligned_xrseq'
        lambda_L1 = 100   ## default is 100, trail_1 is 0.001
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
        
        gan_mode = 'vanilla' # no_lsgan = False
        display_freq = 100
        display_single_pane_ncols =0
        update_html_freq =1000
        print_freq =100
        save_latest_freq = 5000
        save_epoch_freq = 1
        
        continue_train = False
        epoch_count =1
        which_epoch = 'lastest'
        niter = 100
        niter_decay=100
        beta1 = 0.5
        lr = 0.0001
        no_html = False
        lr_policy ='lambda'# plateau
        lr_decay_iters =50
        n_layers_D = 3
        
        
