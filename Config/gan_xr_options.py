import os
import torch

class gan_xr_v1():
#         dataroot = os.path.expanduser('~')+'/Code/XR_project/Data/Sample/'
        dataroot = '../../../../XR_project/Data/1frame/'
        batchSize = 40
        loadSize = 300 # The original value is 286,  The Image size is 
        fineSize = 256
        input_nc = 1
        output_nc = 1
        
        ngf = 32
        ndf = 32

        which_model_netD = 'n_layers' # 'n_layers','piexl'
        n_downsampling = 2  # ADDed By John

        gpu_ids = [0,1]
        name = 'GAN_XR'
        model = 'p2p_xr'
        direction = 'AtoB'
        nThreads = 4
#         checkpoints_dir = os.path.expanduser('~')+'/Code/XR_project/Result/Trial_2/'
        checkpoints_dir = '../../Result/single_25_flip_9/'
        norm = 'instance'
        serial_batches = False
        which_model_netG = 'resnet_9blocks'

        #************** important change ********************
        dataset_mode = 'aligned_xr'
        lambda_L1 = 25   ## default is 100, trail_1 is 0.001
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
        display_freq = 1000
        display_single_pane_ncols =0
        update_html_freq =1000
        print_freq =1000
        save_latest_freq = 50000
        save_epoch_freq =10
        
        continue_train = False
        epoch_count = 1
        which_epoch = 'lastest'
        niter = 100
        niter_decay=100
        beta1 = 0.5
        lr = 0.0002
        no_html = False
        lr_policy ='lambda'
        lr_decay_iters =50
        n_layers_D = 3


class gan_xr_v2():
        name = 'GAN_XR_trail3'
        model = 'p2p_xr'
#         dataroot = os.path.expanduser('~')+'/Code/XR_project/Data/Sample/'
        dataroot = '../../Data/Sample/'
        batchSize = 48
        loadSize = 300 # The original value is 286,  The Image size is 
        fineSize = 256
        input_nc = 1
        output_nc = 1
        
        ngf = 32
        ndf = 32

        which_model_netD = 'n_layers' # 'n_layers','piexl'
        n_downsampling = 2  # ADDed By John

        gpu_ids = [0,1]

        direction = 'AtoB'
        nThreads = 32
#         checkpoints_dir = os.path.expanduser('~')+'/Code/XR_project/Result/Trial_3/'
        checkpoints_dir = '../../Result/Trial_3/'
        norm = 'instance'
        serial_batches = False
        which_model_netG = 'unet_256' #'resnet_6blocks'


        #**************  ********************
        dataset_mode = 'aligned_xr'
        lambda_L1 = 50   ## default is 100, trail_1 is 0.001
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
        save_epoch_freq =10
        
        continue_train = False
        epoch_count =1
        which_epoch = 'lastest'
        niter = 100
        niter_decay=100
        beta1 = 0.5
        lr = 0.0002
        no_html = False
        lr_policy ='lambda'
        lr_decay_iters =50
        n_layers_D = 3

        
        
        
class gan_xr_v4():
        name = 'GAN_XR_trail4'
        model = 'p2p_xr'
#         dataroot = os.path.expanduser('~')+'/Code/XR_project/Data/Sample/'
        dataroot = '../../Data/Sample/'
        
        batchSize = 32
        loadSize = 300 # The original value is 286,  The Image size is 
        fineSize = 256
        input_nc = 1
        output_nc = 1
        
        ngf = 64
        ndf = 64

        which_model_netD = 'n_layers' # 'n_layers','piexl'
        n_downsampling = 2  # ADDed By John

        gpu_ids = [0,1]

        direction = 'AtoB'
        nThreads = 32
#         checkpoints_dir = os.path.expanduser('~')+'/Code/XR_project/Result/Trial_4/'
        checkpoints_dir = '../../Result/Trial_4/'
        norm = 'instance'
        serial_batches = False
        which_model_netG = 'resnet_9block'  #'unet_256' #'resnet_6blocks'


        #**************  ********************
        dataset_mode = 'aligned_xr'
        lambda_L1 = 5   ## default is 100, trail_1 is 0.001
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
        save_epoch_freq =10
        
        continue_train = False
        epoch_count =1
        which_epoch = 'lastest'
        niter = 100
        niter_decay=100
        beta1 = 0.5
        lr = 0.0002
        no_html = False
        lr_policy ='lambda'
        lr_decay_iters =50
        n_layers_D = 3

class gan_xrp_v5():  #极坐标 unet
#         dataroot = os.path.expanduser('~')+'/Code/XR_project/Data/Sample_polar/'
        dataroot = '../../Data/coordinate/'
        batchSize = 64
        loadSize = 256 # The original value is 286,  The Image size is 
        fineSize = 256
        input_nc = 1
        output_nc = 1
        
        ngf = 32
        ndf = 32

        which_model_netD = 'n_layers' # 'n_layers','piexl'
        n_downsampling = 2  # ADDed By John

        gpu_ids = [0,1]
        name = 'GAN_XR_with_polar_transform'
        model = 'p2p_xr'
        direction = 'AtoB'
        nThreads = 13
#         checkpoints_dir = os.path.expanduser('~')+'/Code/XR_project/Result/Trial_5/'
        checkpoints_dir = '../../Result/Trial_5/'
        
        norm = 'instance'
        serial_batches = False
        which_model_netG = 'unet_256'

        #************** important change ********************
        dataset_mode = 'aligned_xr'
        lambda_L1 = 10   ## default is 100, trail_1 is 0.001
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
        save_epoch_freq =10
        
        continue_train = False
        epoch_count =1
        which_epoch = 'lastest'
        niter = 100
        niter_decay=100
        beta1 = 0.5
        lr = 0.0002
        no_html = False
        lr_policy ='lambda'
        lr_decay_iters =50
        n_layers_D = 3