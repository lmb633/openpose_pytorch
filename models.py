import torch
import torch.nn as nn
from torchvision import models


class OpenPose(nn.Module):
    def __init__(self, net_dict, bn=False):
        super(OpenPose, self).__init__()
        self.layer0 = self.make_layer1(net_dict[0], bn, True)
        models1 = []
        models2 = []
        for i in range(1, len(net_dict)):
            models1.append(self.make_layer1(net_dict[i][0], bn))
            models2.append(self.make_layer1(net_dict[i][1], bn))
        self.vec_layers = nn.ModuleList(models1)
        self.heat_layers = nn.ModuleList(models2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer1(self, net_dict, bn=False, last_act=False):
        layer = []
        for i, layer_dict in enumerate(net_dict):
            key = list(layer_dict.keys())[0]
            v = layer_dict[key]
            if 'pool' in key:
                layer.append(nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2]))
            else:
                layer.append(nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]))
                if i < len(net_dict) - 1 or last_act:
                    if bn:
                        layer.append(nn.BatchNorm2d(v[1]))
                    layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)

    def forward(self, x, mask):
        out0 = self.layer0(x)
        pre = out0
        vec_out = []
        heat_out = []
        for vec, heat in zip(self.vec_layers, self.heat_layers):
            out1 = vec(pre)
            out2 = heat(pre)
            pre = torch.cat([out1, out2, out0], 1)
            vec_out.append(out1 * mask)
            heat_out.append(out2 * mask)
        return vec_out, heat_out


def get_model(num_point=19, num_vector=19, num_stages=6, bn=False, pretrained=False):
    net_dict = []
    block0 = [{'conv1_1': [3, 64, 3, 1, 1]}, {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1': [2, 2, 0]},
              {'conv2_1': [64, 128, 3, 1, 1]}, {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2': [2, 2, 0]},
              {'conv3_1': [128, 256, 3, 1, 1]}, {'conv3_2': [256, 256, 3, 1, 1]}, {'conv3_3': [256, 256, 3, 1, 1]}, {'conv3_4': [256, 256, 3, 1, 1]}, {'pool3': [2, 2, 0]},
              {'conv4_1': [256, 512, 3, 1, 1]}, {'conv4_2': [512, 512, 3, 1, 1]}, {'conv4_3_cpm': [512, 256, 3, 1, 1]}, {'conv4_4_cpm': [256, 128, 3, 1, 1]}]
    net_dict.append(block0)

    block1 = [[], []]
    in_vec = [0, 128, 128, 128, 128, 512, num_vector * 2]
    in_heat = [0, 128, 128, 128, 128, 512, num_point]
    for i in range(1, 6):
        if i < 4:
            block1[0].append({'stage1_conv{}_vec'.format(i): [in_vec[i], in_vec[i + 1], 3, 1, 1]})
            block1[1].append({'stage1_conv{}_heat'.format(i): [in_heat[i], in_heat[i + 1], 3, 1, 1]})
        else:
            block1[0].append({'stage1_conv{}_vec'.format(i): [in_vec[i], in_vec[i + 1], 1, 1, 0]})
            block1[1].append({'stage1_conv{}_heat'.format(i): [in_heat[i], in_heat[i + 1], 1, 1, 0]})
    net_dict.append(block1)
    in_vec_1 = [0, 128 + num_point + num_vector * 2, 128, 128, 128, 128, 128, 128, num_vector * 2]
    in_heat_1 = [0, 128 + num_point + num_vector * 2, 128, 128, 128, 128, 128, 128, num_point]
    for j in range(2, num_stages + 1):
        blocks = [[], []]
        for i in range(1, 8):
            if i < 6:
                blocks[0].append({'stage{}conv{}_vec'.format(j, i): [in_vec_1[i], in_vec_1[i + 1], 7, 1, 3]})
                blocks[1].append({'stage{}conv{}_heat'.format(j, i): [in_heat_1[i], in_heat_1[i + 1], 7, 1, 3]})
            else:
                blocks[0].append({'stage{}conv{}_vec'.format(j, i): [in_vec_1[i], in_vec_1[i + 1], 1, 1, 0]})
                blocks[1].append({'stage{}conv{}_heat'.format(j, i): [in_heat_1[i], in_heat_1[i + 1], 1, 1, 0]})
        net_dict.append(blocks)
    model = OpenPose(net_dict, bn)
    model_dict = model.state_dict()

    if pretrained:
        print('use pretrained')
        parameter_num = 10
        if bn:
            vgg19 = models.vgg19_bn(pretrained=True)
            parameter_num *= 6
        else:
            vgg19 = models.vgg19(pretrained=True)
            parameter_num *= 2
        vgg19_state_dict = vgg19.state_dict()
        vgg19_keys = list(vgg19_state_dict.keys())
        model_key = list(model_dict.keys())
        for i in range(parameter_num):
            model_dict[model_key[i]] = vgg19_state_dict[vgg19_keys[i]]
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    model = get_model(bn=False, pretrained=False)
    i = 0
    for k, v in model.state_dict().items():
        print(i, k, v.shape)
        i += 1
    x = torch.zeros((1, 3, 64, 64))
    mask = torch.zeros((1, 1, 8, 8))
    out1, out2 = model(x, mask)
    print(len(out1), len(out2))
    print(out1[0].shape)
    print(out2[0].shape)

    checkpoint = torch.load('BEST_checkpoint.tar', map_location='cpu')
    i = 0
    for k, v in checkpoint['state_dict'].items():
        print(i, k, v.shape)
        i += 1
