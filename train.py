import os
import torch
import torch.nn as nn
from models import get_model
import numpy as np
from data_gen import CocoFolder, file_dir, transfor
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import AverageMeter
from utils import adjust_learning_rate
from config import cfg


def train():
    torch.manual_seed(2)
    np.random.seed(2)
    checkpoint_path = cfg.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epoch_since_improvement = 0

    dataset = CocoFolder(file_dir, cfg.stride, transfor)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = get_model(cfg.num_point, cfg.num_vector, cfg.num_stages, cfg.bn, cfg.pretrained)
    model = torch.nn.DataParallel(model).cuda()
    if os.path.exists(checkpoint_path):
        print('=========load checkpoint========')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.mon, weight_decay=cfg.weight_decay)
    criterion1 = torch.nn.MSELoss().cuda()
    criterion2 = torch.nn.MSELoss(reduction='none').cuda()

    log = SummaryWriter(log_dir='data/log', comment='openpose')
    for epoch in range(start_epoch, cfg.epoch):
        adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        loss = train_once(model, train_loader, optimizer, [criterion1, criterion2], epoch, log)
        log.add_scalar('cpn_loss', loss, epoch)
        log.flush()
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optm': optimizer,
            }, checkpoint_path)
            epoch_since_improvement = 0
        else:
            epoch_since_improvement += 1
    log.close()


def train_once(model, train_loader, optimizer, criterion, epoch, log):
    model.train()
    losses = AverageMeter()
    heat_weight = 46 * 46 * 19 / 2.0  # for convenient to compare with origin code
    vec_weight = 46 * 46 * 38 / 2.0
    for i, (inputs, heatmap, vecmap, mask) in enumerate(train_loader):
        inputs = inputs.cuda()
        mask = mask.cuda()
        # print(inputs.shape)
        # print(len(targets), print(targets[-1].shape))
        vec_pred, heat_pred = model(inputs, mask)

        vec_loss = 0
        heat_loss = 0
        for vec in vec_pred:
            vec_loss += criterion(vec, vecmap) * vec_weight
        for heat in heat_pred:
            heat_loss += criterion(heat, heatmap) * heat_weight
        loss = vec_loss + heat_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        log.add_scalar('loss_epoch_{0}'.format(epoch), loss.item(), i)
        log.flush()
        if i % cfg.print_freq == 0:
            print('epoch: ', epoch, '{0}/{1} loss_avg: {2} global_loss: {3} refine_loss: {4} loss: {5}'.format(i, len(train_loader), losses.avg, vec_loss, heat_loss, loss))
    return losses.avg


if __name__ == '__main__':
    train()
