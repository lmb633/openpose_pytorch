import torch
import torch.utils.data as data
import numpy as np
import shutil
import time
import random
import os
import math
import json
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import mytransforms

image_dir = 'data/train2017/'
mask_dir = 'data/mask/'
file_path = 'data/data_json/train/filelist.txt'
mask_path = 'data/data_json/train/masklist.txt'
json_path = 'data/data_json/train/train2017.json'

file_dir = [file_path, mask_path, json_path]
transfor = mytransforms.Compose([mytransforms.RandomResized(),
                                 mytransforms.RandomRotate(40),
                                 mytransforms.RandomCrop(368),
                                 mytransforms.RandomHorizontalFlip(),
                                 ])


def read_data_file(file_dir, root_dir):
    lists = []
    with open(file_dir, 'r') as fp:
        line = fp.readline()
        while line:
            path = line.strip()
            lists.append(root_dir + path.split('/')[-1])
            line = fp.readline()

    return lists


def read_json_file(file_dir):
    """
        filename: JSON file

        return: two list: key_points list and centers list
    """
    fp = open(file_dir)
    data = json.load(fp)
    kpts = []
    centers = []
    scales = []

    for info in data:
        kpt = []
        center = []
        scale = []
        lists = info['info']
        for x in lists:
            kpt.append(x['keypoints'])
            center.append(x['pos'])
            scale.append(x['scale'])
        kpts.append(kpt)
        centers.append(center)
        scales.append(scale)
    fp.close()

    return kpts, centers, scales


def generate_heatmap(heatmap, kpt, stride, sigma):
    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] > 1:
                continue
            x = kpt[i][j][0]
            y = kpt[i][j][1]
            for h in range(height):
                for w in range(width):
                    xx = start + w * stride
                    yy = start + h * stride
                    dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                    if dis > 4.6052:
                        continue
                    heatmap[h][w][j + 1] += math.exp(-dis)
                    if heatmap[h][w][j + 1] > 1:
                        heatmap[h][w][j + 1] = 1

    return heatmap


def generate_vector(vector, cnt, kpts, vec_pair, stride, theta):
    height, width, channel = cnt.shape
    # print(vector.shape, cnt.shape)
    length = len(kpts)
    for j in range(length):
        for i in range(channel):
            a = vec_pair[0][i]
            b = vec_pair[1][i]
            # print(a, b)
            if kpts[j][a][2] > 1 or kpts[j][b][2] > 1:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9  # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba
            # print(bax, bay)

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)
            # print(min_w, max_w, min_h, max_h)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay
                    # print(px, py)
                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1
                        # print(vector[min_h:max_h, min_w:max_w, 0])
                        # print(vector[min_h:max_h, min_w:max_w, 1])
                        # print(cnt[min_h:max_h, min_w:max_w, 0])

    return vector


class CocoFolder(data.Dataset):
    def __init__(self, file_dir, stride, transformer=None):
        self.img_list = read_data_file(file_dir[0], image_dir)
        self.mask_list = read_data_file(file_dir[1], mask_dir)
        self.kpt_list, self.center_list, self.scale_list = read_json_file(file_dir[2])
        self.stride = stride
        self.transformer = transformer
        self.vec_pair = [[2, 3, 5, 6, 8, 9, 11, 12, 0, 1, 1, 1, 1, 2, 5, 0, 0, 14, 15],
                         [3, 4, 6, 7, 9, 10, 12, 13, 1, 8, 11, 2, 5, 16, 17, 14, 15, 16, 17]]  # different from openpose
        self.theta = 1.0
        self.sigma = 7.0

    def __getitem__(self, index):
        img_path = self.img_list[index]

        img = np.array(cv2.imread(img_path), dtype=np.float32)
        mask_path = self.mask_list[index]
        mask = np.load(mask_path)
        mask = np.array(mask, dtype=np.float32)

        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]
        img, mask, kpt, center = self.transformer(img, mask, kpt, center, scale)
        height, width, _ = img.shape

        mask = cv2.resize(mask, (width // self.stride, height // self.stride)).reshape((height // self.stride, width // self.stride, 1))

        heatmap = np.zeros((height // self.stride, width // self.stride, len(kpt[0]) + 1), dtype=np.float32)
        heatmap = generate_heatmap(heatmap, kpt, self.stride, self.sigma)
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background
        heatmap = heatmap * mask

        vecmap = np.zeros((height // self.stride, width // self.stride, len(self.vec_pair[0]) * 2), dtype=np.float32)
        cnt = np.zeros((height // self.stride, width // self.stride, len(self.vec_pair[0])), dtype=np.int32)
        vecmap = generate_vector(vecmap, cnt, kpt, self.vec_pair, self.stride, self.theta)
        vecmap = vecmap * mask

        img = mytransforms.normalize(mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0])  # mean, std
        mask = mytransforms.to_tensor(mask)
        heatmap = mytransforms.to_tensor(heatmap)
        vecmap = mytransforms.to_tensor(vecmap)

        return img, heatmap, vecmap, mask

    def __len__(self):
        return len(self.img_list)


# ["nose", "neck", "right_shoulder","right_elbow","right_wrist","left_shoulder","left_elbow","left_wrist","right_hip","right_knee","right_ankle" "left_hip","left_knee",,"left_ankle,
#   "right_eye","left_eye","right_ear","left_ear"],
#	"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
def vis():
    coco = CocoFolder(file_dir, 8, mytransforms.Compose([mytransforms.RandomResized(),
                                                         mytransforms.RandomRotate(40),
                                                         mytransforms.RandomCrop(368),
                                                         mytransforms.RandomHorizontalFlip(),
                                                         ]))

    image, heatmaps, vecmap, mask = coco[9]
    print(image.shape, heatmaps.shape, vecmap.shape, mask.shape)
    image = image.numpy().transpose(1, 2, 0)
    image *= 255
    image += 128
    image /= 255
    print(mask.shape)
    # for line in mask[0]:
    #     print(line)
    mask = mask.numpy().transpose(1, 2, 0)
    mask = cv2.resize(mask, (368, 368))
    mask = mask.reshape((368, 368))
    # new_img = image * mask
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.show()
    plt.close()

    heatmaps = heatmaps.numpy().transpose(1, 2, 0)
    heatmaps = cv2.resize(heatmaps, (368, 368))
    for j in range(19):
        print(j)
        heatmap = heatmaps[:, :, j]
        heatmap = heatmap.reshape((368, 368))
        heatmap *= 255
        heatmap /= 255
        # result = heatmap * 0.4 + img * 0.5
        # plt.imshow(image)
        # plt.imshow(heatmap, alpha=0.5)
        # plt.show()
        # plt.close()
    print(vecmap.shape)
    vecs = vecmap.numpy()
    vecs = vecs.transpose(1, 2, 0)
    vecs = cv2.resize(vecs, (368, 368))

    for j in range(0, 38):
        vec = np.abs(vecs[:, :, j])
        # vec += np.abs(vecs[:, :, j + 1])
        vec[vec > 1] = 1
        vec = vec.reshape((368, 368, 1))
        # vec[vec > 0] = 1
        vec *= 255
        # vec = cv2.applyColorMap(vec, cv2.COLORMAP_JET)
        vec = vec.reshape((368, 368))
        vec /= 255

        plt.imshow(image)
        # result = vec * 0.4 + img * 0.5
        plt.imshow(vec, alpha=0.5)
        plt.show()
        plt.close()


def vis_mask():
    mask_list = read_data_file(mask_path, mask_dir)
    print(mask_list)
    # for mask in mask_list:
    #     print(mask)
    mask = np.load('data/mask/000000463730.npy').astype(dtype=np.float)
    image = cv2.imread('data/val2017/000000463730.jpg')
    image = image.astype(dtype=np.float)
    image /= 255
    print(image.shape)
    print(mask.shape)
    # cv2.imshow('',image)
    # cv2.waitKey(-1)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.show()
    plt.close()


if __name__ == '__main__':
    vis()
