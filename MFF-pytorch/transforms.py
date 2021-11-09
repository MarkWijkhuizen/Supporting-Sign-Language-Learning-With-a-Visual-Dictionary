import warnings
warnings.simplefilter("ignore", UserWarning)

from PIL import Image, ImageOps
from cv2 import resize
from tqdm import tqdm

import numpy as np

import torchvision
import random
import cv2
import numbers
import math
import torch
import time
import psutil
import sys

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, rescale, mean, std, n_frames, num_segments, num_motion, debug=False):
        self.rescale = rescale
        self.mean = mean
        self.std = std
        self.debug = debug

        self.rep_mean = list(mean) + ([np.mean(mean)] * 2 * num_motion)
        self.rep_mean = np.array(self.rep_mean * num_segments, dtype=np.float32)
        self.rep_mean = self.rep_mean[:, np.newaxis, np.newaxis]
        
        self.rep_std = list(std) + ([np.mean(std)] * 2 * num_motion)
        self.rep_std = np.array(self.rep_std * num_segments, dtype=np.float32)
        self.rep_std = self.rep_std[:, np.newaxis, np.newaxis]

    def __call__(self, imgs):
        if self.debug:
            t_start = time.time()

        # normalize to value between (0,1)
        imgs = imgs.astype(np.float32) / self.rescale
        # Normalize to mean 0 std 1 for imagenet
        imgs = (imgs - self.rep_mean) / self.rep_std

        if self.debug:
            print(f'GroupNormalize took: {int((time.time() - t_start) * 1000)}ms')

        return imgs

class ToTorch(object):
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self, imgs):
        return torch.tensor(imgs)


class GroupNormalizeOriginal(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(True, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                #print(img.size)
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                #print([o_w, o_h, o_w + crop_w, o_h + crop_h])
                normal_group.append(crop)
                #flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)
                #flip_group.append(flip_crop)

                #if img.mode == 'L' and i % 2 == 0:
                    #flip_group.append(ImageOps.invert(flip_crop))
                #else:
                    #flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            #oversample_group.extend(flip_group)
        return oversample_group


class GroupSpatialElasticDisplacement(object):

    def __init__(self):
        self.displacement = 20
        self.displacement_kernel = 25
        self.displacement_magnification = 0.60


    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            im_size = img_group[0].size
            image_w, image_h = im_size[0], im_size[1]
            displacement_map = np.random.rand(image_h, image_w, 2) * 2 * self.displacement - self.displacement
            displacement_map = cv2.GaussianBlur(displacement_map, None, self.displacement_kernel)
            displacement_map *= self.displacement_magnification * self.displacement_kernel
            displacement_map = np.floor(displacement_map).astype('int32')

            displacement_map_rows = displacement_map[..., 0] + np.tile(np.arange(image_h), (image_w, 1)).T.astype('int32')
            displacement_map_rows = np.clip(displacement_map_rows, 0, image_h - 1)

            displacement_map_cols = displacement_map[..., 1] + np.tile(np.arange(image_w), (image_h, 1)).astype('int32')
            displacement_map_cols = np.clip(displacement_map_cols, 0, image_w - 1)
            ret_img_group = [Image.fromarray(np.asarray(img)[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(np.asarray(img).shape)) for img in img_group]
            return ret_img_group

        else:
            return img_group


class GroupMultiScaleResize(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img_group):
        im_size = img_group[0].size
        self.resize_const = random.uniform(1.0 - self.scale, 1.0 + self.scale) # Aplly random resize constant 
        resize_img_group = [img.resize((int(im_size[0]*self.resize_const), int(im_size[1]*self.resize_const))) for img in img_group]

        return resize_img_group


class GroupMultiScaleRotate(object):

    def __init__(self, degree, mffs_shape, resize_only=False, debug=False):
        self.degree = degree
        self.mffs_shape = mffs_shape
        self.resize_shape = tuple(mffs_shape[-2:])
        self.interpolation = Image.BILINEAR
        self.resize_only = resize_only
        self.debug = debug

    def __call__(self, imgs):
        if self.debug:
            t_start = time.time()

        rotate_angle = random.randint(-self.degree, self.degree) # Aplly random rotation angle

        res = np.empty(shape=self.mffs_shape, dtype=np.float32)
        for img_idx, img in enumerate(imgs):
            # Only Resize Image
            if self.resize_only:
                res[img_idx] = Image.fromarray(img).resize(self.resize_shape, resample=Image.BILINEAR)
            # Rotate and Resize
            else:
                res[img_idx] = Image.fromarray(img).rotate(rotate_angle, fillcolor=0, resample=Image.BILINEAR).resize(self.resize_shape, resample=Image.BILINEAR)

        if self.debug:
            print(f'GroupMultiScaleRotate took: {int((time.time() - t_start) * 1000)}ms')

        return res



class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=False):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size
        #self.scales = [1, random.uniform(0.85, 1.0)]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]

        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0 * w_step, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False, isRGBFlow=False):
        self.roll = roll
        self.isRGBFlow = isRGBFlow

    def __call__(self, img_group):
        if self.isRGBFlow:
            stacked_array = np.array([])
            for x in img_group:
                if x.mode == 'L':
                    if stacked_array.size ==0:
                        stacked_array = np.expand_dims(x, 2)
                    else:
                        stacked_array = np.concatenate([stacked_array, np.expand_dims(x, 2)], axis=2)
                elif x.mode == 'RGB':
                    if self.roll:
                        stacked_array = np.concatenate([stacked_array, np.array(x)[:, :, ::-1]], axis=2)
                    else:
                        stacked_array = np.concatenate([stacked_array, np.array(x)], axis=2)
            return stacked_array
          
        else:
            if img_group[0].mode == 'L':
                return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
            elif img_group[0].mode == 'RGB':
                if self.roll:
                    asd = np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
                    return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
                else:
                    return np.concatenate(img_group, axis=2)


class StackFast(object):
    def __init__(self, IMG_SIZE, N_MFF_FRAMES, N_FRAMES):
        self.IMG_SIZE = IMG_SIZE
        self.N_MFF_FRAMES = N_MFF_FRAMES
        self.N_FRAMES = N_FRAMES

    def __call__(self, img_group):
        # create empty result array
        res = np.empty(shape=[self.IMG_SIZE, self.IMG_SIZE, self.N_MFF_FRAMES * self.N_FRAMES], dtype=np.uint8)
        frame_counter = 0
        for img in img_group:
            if img.mode == 'RGB':
                rgb_frame = np.array(img)
                # flip frame
                rgb_frame = np.flip(rgb_frame, axis=2)
                res[:,:,frame_counter:frame_counter+3] = rgb_frame
                frame_counter += 3
            else:
                res[:,:,frame_counter:frame_counter+1] = np.expand_dims(np.array(img), axis=2)
                frame_counter += 1

        return res


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class IdentityTransform(object):

    def __call__(self, data):
        return data

# Own Crop/Resize Function
class CropPad(object):

    def __init__(self, scale, img_size, debug=False):
        self.scale = scale
        self.img_size = img_size
        self.debug = debug
    
    def __call__(self, imgs):
        if self.debug:
            t_start = time.time()

        s = self.img_size
        res = []
        # random scale
        scale = np.random.randint(-self.scale, self.scale+1) / 100
        # crop/pad each image
        crop_size = int(s * (1 + scale))

        if scale < 0:
            crop_size = int(s * (1 + scale))
            x_offset = np.random.randint(0, s - crop_size)
            y_offset = np.random.randint(0, s - crop_size)
            imgs = imgs[:, y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
        else:
            pad_size = int(s * (1 + scale)) - s
            imgs = np.pad(imgs, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

        if self.debug:
            print(f'CropPad took: {int((time.time() - t_start) * 1000)}ms')

        return imgs

class GridMask(object):

    def __init__(self, mff_frames, debug=False):
        self.debug = debug
        self.mff_frames = mff_frames

    def __call__(self, imgs):
        if self.debug:
            t_start = time.time()

        _, _, img_size = imgs.shape
        d = np.random.randint(96, 224)
        grid = np.array([[[0], [1]],[[1], [0]]], dtype=np.float32)
        grid = resize(grid, (d, d), interpolation=cv2.INTER_NEAREST)
        
        # 50% chance to rotate mask
        if np.random.randint(1) == 0:
            grid = np.rot90(grid)

        repeats = img_size // d + 1
        grid = np.tile(grid, reps=[repeats, repeats])
        grid_size, _ = grid.shape

        grid_batch = np.empty(shape=imgs.shape, dtype=np.float32)
        for idx in range(len(imgs)):
            if idx % self.mff_frames < self.mff_frames - 3:
                offset_x = np.random.randint(0, grid_size - img_size)
                offset_y = np.random.randint(0, grid_size - img_size)
            
            grid_frame = grid[offset_x:offset_x+img_size, offset_y:offset_y+img_size]
            grid_batch[idx] = grid_frame

        imgs = imgs * grid_batch

        if self.debug:
            print(f'GridMask took: {int((time.time() - t_start) * 1000)}ms')

        return imgs

if __name__ == "__main__":

    from dataset_custom import TSNDataSet

    N_FRAMES = (3 + 5 * 2) * 4

    workers = 0 #psutil.cpu_count(logical=True)
    print(f'Using {workers} workers')

    img_lena = np.array(Image.open('lena.png').convert('L'))
    
    # img_lena = np.array(Image.open('C:/Users/markw/Downloads/ngt_10fps_val/flow/x/166/00001.jpg'))
    # print(np.median(img_lena))

    print('MAIN', np.array(img_lena).shape)
    mff_lena = np.expand_dims(img_lena, 0)
    mff_lena = np.repeat(mff_lena, repeats=N_FRAMES, axis=0)

    print('MAIN len mff', mff_lena.shape)

    debug = True
    transforms = torchvision.transforms.Compose([
        GroupNormalize(255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], N_FRAMES, 4, 5, debug=debug),
        # CropPad(20, 224, debug=debug),
        # GroupMultiScaleRotate(20, (N_FRAMES,224,224), resize_only=False, debug=debug),
        # GridMask(9, debug=debug),
    ])

    dataset = torch.utils.data.DataLoader(
        TSNDataSet('jester_8-MFFs-3f1c_of_kfe_v2_fr2_train', (N_FRAMES,224,224), transforms, debug=True, n=100, compression='lzma'),
        batch_size=10,
        shuffle=True,
        num_workers=0,
    )

    t_start = time.time()
    for step, (mffs, lbls) in enumerate(iter(dataset)):
        for i in range(4):
            
            img = mffs.numpy()[0,i*9+6:i*9+6+3]
            print('original', img.shape)
            img = np.moveaxis(img, 0, 2)
            if img.shape[2] == 1:
                img = img[:,:,0]
                img = (img * 0.229) + 0.485
            else:
                img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            
            img = (img * 255).astype(np.uint8)
            # Image.fromarray(img).show()
            print(f'img shape: {img.shape}')
            break
        break

    print(f'mffs shape: {mffs.shape}, mffs dtype: {mffs.dtype}, lbls shape: {lbls.shape}')
    mffs = np.array(mffs, dtype=np.float32)
    print(f'mffs mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s' % (mffs.mean(), mffs.std(), mffs.min(), mffs.max(), mffs.dtype))
    print(f'Transforms took: {int((time.time() - t_start) * 1000)}ms')