class Config:
    checkpoint = 'BEST_checkpoint.tar'
    batch_size = 64
    weight_decay = 5e-4
    mon = 0.9
    epoch = 1000
    print_freq = 10

    lr = 4e-5
    lr_gamma = 0.5
    lr_dec_epoch = list(range(6, 40, 6))

    num_point = 19
    num_vector = 19
    num_stages = 6
    stride = 8
    bn = False
    pretrained = False


cfg = Config()
