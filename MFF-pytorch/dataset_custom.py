import torch.utils.data as data
import numpy as np

from PIL import Image
from tqdm import tqdm

import pickle
import glob
import os
import bz2
import time
import io
import sys
import gc
import lzma

class TSNDataSet(data.Dataset):
    def __init__(
            self, subset, mffs_shape_dataset, mffs_shape_model, transforms, is_train=True, debug=False, n=None,
            index_label=False, compression=None, dataset_ratio=1.00, cache=False, attr=[], num_class=-1,
            multi_sim_label=False, num_motion_subset=None,  mff_frames_dataset=None,
        ):

        print('subset', subset)
        # Compression file format
        if compression == 'lzma':
            print('Looking for lzma compressed .xr files')
            self.pickle_file_paths = glob.glob(os.path.join('datasets', 'pickle', subset, '*.xr'))
        else:
            print('Looking for uncompressed .pkl files')
            self.pickle_file_paths = glob.glob(os.path.join('datasets', 'pickle', subset, '*.pkl'))

        self.pickle_folder = os.path.join('datasets', 'pickle', subset)
        # use subset of pickles
        if n is not None:
            print(f'Using subset of {n} images')
            self.pickle_file_paths = self.pickle_file_paths[:n]
        
        if dataset_ratio != 100:
            len_original = len(self.pickle_file_paths)
            self.pickle_file_paths = self.pickle_file_paths[:int(len(self.pickle_file_paths) * dataset_ratio)]
            len_ratio = len(self.pickle_file_paths)
            print(f'Using {dataset_ratio * 100:.2f}%({len_ratio}/{len_original}) images of dataset')

        self.dataset_size = len(self.pickle_file_paths)
        self.transforms = transforms
        self.is_train = is_train
        self.mffs_shape_dataset = mffs_shape_dataset
        self.mffs_shape_model = mffs_shape_model
        self.debug = debug
        self.index_label = index_label
        self.compression = compression
        self.num_class = num_class
        self.attr = attr
        self.multi_sim_label = multi_sim_label
        self.num_motion_subset = num_motion_subset
        self.mff_frames_dataset = mff_frames_dataset
        # Cache Mechanism
        self.cache = cache
        if self.cache:
            print('Caching dataset')
            self.index_cached = np.full(self.dataset_size, False)
            self.imgs_square_cache = np.empty(shape=[self.dataset_size, *self.mffs_shape_model], dtype=np.uint8)
            if multi_sim_label:
                self.label_cache = np.empty(shape=[self.dataset_size, 5, self.num_class], dtype=np.float64)
            else:
                self.label_cache = np.empty(shape=[self.dataset_size], dtype=np.int64)
            # Attributes cache
            self.attr_cache = np.full(self.dataset_size, object)
            for idx, _ in enumerate(self.attr_cache):
                self.attr_cache[idx] = dict()
        
        print(f'Found {self.dataset_size} Records in {subset} Folder')

        # Print Augmentations
        print(f'-- {subset.upper()} AUGMENTATIONS')
        for t in transforms.transforms:
            print(f'* {t.__class__.__name__}')

    # Load pickle with given compression
    def load_pickle(self, file_path):
        if self.compression == None:
            with open(file_path, 'rb') as f:
                record = pickle.load(f)
        elif self.compression == 'bz2':
            with bz2.BZ2File(file_path, 'rb') as f:
                record = pickle.load(f)
        elif self.compression == 'lzma':
            with lzma.LZMAFile(file_path, 'rb') as f:
                record = pickle.load(f)

        return record

    def __getitem__(self, index, file_path=None):
        if self.debug:
            t_start_total = time.time()
            t_start = time.time()
        
        if self.cache and self.index_cached[index]:
            imgs_square = self.imgs_square_cache[index]
            label = self.label_cache[index]
            attr = dict()
            for a in self.attr:
                attr[a] = self.attr_cache[index][a]
        else:
            imgs_square, label, attr = self.get_imgs_square_label(index, file_path)
        
        if self.debug:
            print(f'Image reading took: {int((time.time() - t_start) * 1000)}ms')
            t_start = time.time()
        
        mffs = self.transforms(imgs_square)
        
        if self.debug:
            print(f'Transofmrations took: {int((time.time() - t_start) * 1000)}ms')
            print(f'TOTAL MFF READING TIME took: {int((time.time() - t_start_total) * 1000)}ms')
            print('='*50)

        if self.index_label:
            if file_path is None:
                label = index
            else:
                label = self.pickle_file_paths.index(file_path)

        return mffs, label, attr

    def get_imgs_square_label(self, index, file_path):
        if file_path is None:
            imgs, attributes = self.load_pickle(self.pickle_file_paths[index])
        else:
            imgs, attributes = self.load_pickle(file_path)

        if self.debug:
            print(f'Pickle reading took: {int((time.time() - t_start) * 1000)}ms')
            t_start = time.time()

        imgs_square = np.empty(shape=self.mffs_shape_dataset, dtype=np.uint8)

        frame_idx = 0
        offset = None

        for mff_idx, frame_options in enumerate(imgs):
            # random frame idx when training
            if self.is_train:
                frame_option_idx = np.random.randint(0, len(frame_options))
            # linear frame idx for val/test
            else: 
                frame_option_idx = len(frame_options) // 2

            mff = frame_options[frame_option_idx]

            for img_idx, img in enumerate(mff):
                img_pil = Image.open(io.BytesIO(img))
                img_np = np.array(img_pil)
                h, w = img_np.shape[:2]
                n_channels = 3 if len(img_np.shape) == 3 else 1
                # crop if width allows it
                if w > h:
                    if offset is None:
                        # set single offset for MFFS
                        if self.is_train:
                            offset = np.random.randint(0, w-h)
                        else:
                            offset = (w-h) // 2
                    img_np = img_np[:, offset:offset+h]
                # move axis of RGB images
                if n_channels > 1:
                    img_np = np.moveaxis(img_np, 2 ,0)

                # add to result
                imgs_square[frame_idx:frame_idx+n_channels] = img_np
                
                # increase frame index
                frame_idx += 3 if len(img_np.shape) == 3 else 1

        # Optical Flow Frame Subset
        if self.num_motion_subset is not None:
            keep_idxs = []
            for idx in range(len(imgs_square)):
                if idx % self.mff_frames_dataset >= self.mff_frames_dataset - 3 - self.num_motion_subset * 2:
                    keep_idxs.append(idx)
            # Select frames
            imgs_square = imgs_square[keep_idxs]

        # Cache
        if self.cache and not self.index_cached[index] :
            self.index_cached[index] = True
            self.imgs_square_cache[index] = imgs_square
            self.label_cache[index] = attributes['label']
            for k, v in attributes.items():
                self.attr_cache[index][k] = v

        if type(attributes) is dict:
            # Attributes
            attr = dict()
            for a in self.attr:
                attr[a] = attributes[a]

            if self.multi_sim_label:
                label = []
                for key in ['handedness_similarity', 'location_similarity', 'movement_similarity', 'strong_hand_similarity', 'weak_hand_similarity']:
                    label.append(attributes[key])
                label = np.array(label, dtype=np.float64)
            else:
                label = attributes['label']

            return imgs_square, label, attr
        else:
            return imgs_square, attributes, dict()

    def get_record_by_folder_id(self, folder_id):
        if self.compression == 'lzma':
            file_path = os.path.join(self.pickle_folder, f'{folder_id}.xr')
        else:
            file_path = os.path.join(self.pickle_folder, f'{folder_id}.pkl')
        return self.__getitem__(None, file_path)

    def set_test_augs(self):
        self.transforms = torchvision.transforms.Compose([
            # Normalize according to ImageNet means and std
            GroupNormalize(input_rescale, input_mean, input_std, args.n_frames, args.num_segments, args.num_motion),
        ])
        self.is_train = False

    def __len__(self):
        return self.dataset_size