class Config:
    checkpoint = 'BEST_checkpoint.tar'
    batch_size = 64
    weight_decay = 1e-5
    mon = 0.0
    epoch = 1000
    print_freq = 10

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = list(range(6, 40, 6))

    num_point = 19
    num_vector = 19
    num_stages = 6
    stride = 8
    bn = False
    pretrained = False


cfg = Config()
