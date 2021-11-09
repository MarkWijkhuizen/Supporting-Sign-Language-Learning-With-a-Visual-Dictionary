from PIL import Image
from numpy.random import randint

import torch.utils.data as data
import numpy as np
import pandas as pd

import os.path
import os
import glob
import random
import time

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False, dataset_type=None, dataset='jester'):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.dataset = dataset
        self.dataset_type = dataset_type

        if self.modality == 'RGBDiff' or self.modality == 'RGBFlow':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx, isLast=False):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, "rgb", directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, "rgb", directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, "rgb", directory, self.image_tmpl.format(1))).convert('RGB')]
            
        elif self.modality == 'Flow':
            try:
                x_img = Image.open(os.path.join(self.root_path, os.path.join('flow', 'x'), directory, self.image_tmpl.format(idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, os.path.join('flow', 'y'), directory, self.image_tmpl.format(idx))).convert('L')
            except Exception:
                print('error loading flow file:', os.path.join(self.root_path, "flow/v", directory, self.image_tmpl.format(idx)))
                x_img = Image.open(os.path.join(self.root_path, os.path.join('flow', 'x'), directory, self.image_tmpl.format(1))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, os.path.join('flow', 'y'), directory, self.image_tmpl.format(1))).convert('L')
            return [x_img, y_img]

        elif self.modality == 'RGBFlow':
            if isLast:
                return [Image.open(os.path.join(self.root_path, "rgb", directory, self.image_tmpl.format(idx))).convert('RGB')]
            else:
                x_img = Image.open(os.path.join(self.root_path, os.path.join('flow', 'x'), directory, self.image_tmpl.format(idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, os.path.join('flow', 'y'), directory, self.image_tmpl.format(idx))).convert('L')
                return [x_img, y_img]


    def _parse_list(self):
        # superhacky way of reading all precomputed validation images
        if self.dataset_type == 'val':
            folder_names = pd.read_csv('D:/MEGA/Nijmegen/Master Stage/notebooks/MFF OLD/datasets/jester-v1/jester-v1-validation.csv',
                sep=';',
                header=None,
                usecols=[0],
            )[0].astype('string').tolist()
        else:
            folder_names = [f.split('\\')[-1] for f in glob.glob(f'C:/Users/markw/Downloads/20bn-jester-v1/rgb/*')]
        
        list.sort(folder_names)

        folder_names_computed = []
        for idx, _ in enumerate(folder_names):
            if idx < len (folder_names) - 1:
                if os.path.exists(f'C:/Users/markw/Downloads/20bn-jester-v1/flow/x/{folder_names[int(idx)+1]}'):
                    folder_names_computed.append(folder_names[idx])
                else:
                    break
        
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3 and item[0] in folder_names_computed]        
        self.video_list = [VideoRecord(item) for item in tmp]
        print(f'Found {len(self.video_list)} {self.dataset_type} videos')

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        if self.modality == 'RGBFlow':
            while not os.path.exists(os.path.join(self.root_path, "rgb", record.path, self.image_tmpl.format(1))):
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
        else:
            while not os.path.exists(os.path.join(self.root_path, "rgb", record.path, self.image_tmpl.format(1))):
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        t_start = time.time()
        print(f'record: {record}, indices: {indices}')
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                if self.modality == 'RGBFlow':
                    if i == self.new_length - 1:
                        seg_imgs = self._load_image(record.path, p, True)
                    else:
                        if p == record.num_frames:
                            seg_imgs = self._load_image(record.path, p-1)
                        else:
                            seg_imgs = self._load_image(record.path, p)
                else:
                    seg_imgs = self._load_image(record.path, p)

                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        print(f'took: {int(1000 * (time.time() - t_start))}ms')
        print(f'len images: {len(images)}')
        print(f'image 0: {images[0]}')
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
