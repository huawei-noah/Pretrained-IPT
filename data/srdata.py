# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os
import glob
import random
import pickle
import io

import PIL.Image as pil_image
from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
import torchvision.transforms as tfs

def search(root, target="JPEG"):
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            #print('[-]', path)
            item_list.extend(search(path, target))
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
        elif path.split('/')[-2] == target or path.split('/')[-3] == target or path.split('/')[-4] == target:
            item_list.append(path)
        else:
            ttt = 1
            #print('[!]', path)
    return item_list

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        
        if self.args.derain:
            self.derain_test = os.path.join(args.dir_data, "Rain100L")
            self.derain_lr_test = search(self.derain_test, "rain")
            self.derain_hr_test = [path.replace("rainy/","no") for path in self.derain_lr_test]
        if self.args.deblur:
            self.deblur_dataroot = os.path.join(args.dir_data, "GOPRO_Large/train")
            self.deblur_hr = search(self.deblur_dataroot, "sharp")
            self.deblur_lr = search(self.deblur_dataroot, "blur")
            self.deblur_test = os.path.join(args.dir_data, "GOPRO_Large/test")
            self.deblur_hr_test = search(self.deblur_test, "sharp")
            self.deblur_lr_test = search(self.deblur_test, "blur")

        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                if s == 1:
                    os.makedirs(
                        os.path.join(self.dir_hr),
                        exist_ok=True
                    )
                else:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                        ),
                        exist_ok=True
                    )
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) 
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
        if self.args.derain:
            self.images_hr, self.images_lr = self.derain_hr_test, self.derain_lr_test
        if self.args.denoise:
            self.images_hr = glob.glob(os.path.join(self.apath, '*.png'))
            self.images_lr = glob.glob(os.path.join(self.apath, '*.png'))

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                if s != 1:
                    names_lr[si].append(os.path.join(
                        self.dir_lr, 'X{}/{}x{}{}'.format(
                            s, filename, s, self.ext[1]
                        )
                    ))
        for si, s in enumerate(self.scale):
            if s == 1:
                names_lr[si]=names_hr
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        #if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        if self.args.deblur:
            lr, hr, filename = self._load_file_deblur(idx, False)
            pair = self.get_patch(lr, hr)
            pair = common.set_channel(*pair, n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return pair_t[0], pair_t[1], filename
        if self.args.derain:
            norain, rain, filename = self._load_rain_test(idx)
            pair = common.set_channel(*[norain, rain], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return pair_t[0], pair_t[1], filename
        if self.args.denoise:
            hr, filename = self._load_file_hr(idx)
            pair = self.get_patch_hr(hr)
            pair = common.set_channel(*[pair], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return pair_t[0],pair_t[0], filename
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            if self.args.derain:
                return int(len(self.derain_hr_test)/self.args.derain_test)
            if self.args.deblur:
                return int(len(self.deblur_hr_test)/self.args.deblur_test)
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            #if self.args.deblur:
            #    return random.randint(0, len(self.deblur_hr_test) - 1)
            #if self.args.dehaze:
            #    return random.randint(0, len(self.haze_test) - 1)
            return idx

    def _load_file_hr(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)

        return hr, filename
    
    def _load_rain_test(self, idx):
        f_hr = self.derain_hr_test[idx]
        f_lr = self.derain_lr_test[idx]
        filename, _ = os.path.splitext(os.path.basename(f_lr))
        norain = imageio.imread(f_hr)
        rain = imageio.imread(f_lr)
        return norain, rain, filename
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
       
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename
    
    def _load_file_deblur(self, idx, train = True):
        idx = self._get_index(idx)
        if train:
            f_hr = self.images_hr[idx]
            f_lr = self.images_lr[idx]
        else:
            f_hr = self.deblur_hr_test[idx]
            f_lr = self.deblur_lr_test[idx]
        
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        filename = f_hr[-27:-17] + filename
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)

        return lr, hr, filename

    def get_patch_hr(self, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            hr = self.get_patch_img_hr(
                hr,
                patch_size=self.args.patch_size,
                scale=1
            )

        return hr

    def get_patch_img_hr(self, img, patch_size=96, scale=2):
        ih, iw = img.shape[:2]

        tp = patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        ret = img[iy:iy + ip, ix:ix + ip, :]

        return ret

    
    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size*scale,
                scale=scale,
                multi=(len(self.scale) > 1)
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

