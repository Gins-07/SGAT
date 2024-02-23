def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'p2p_xr':
        from .gan_xr import GAN_XR_Model
        model =  GAN_XR_Model()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
