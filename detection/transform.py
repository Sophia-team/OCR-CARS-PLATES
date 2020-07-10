import cv2
import numpy as np
import albumentations as albu

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Pad(object):

    def __init__(self, max_size=1.0, p=0.1):
        self.max_size = max_size
        self.p = p

    def __call__(self, image, mask):
        if np.random.uniform(0.0, 1.0) > self.p:
            return image, mask
        h, w, _ = image.shape
        size = int(np.random.uniform(0, self.max_size) * min(w, h))
        image_ = cv2.copyMakeBorder(image, size, size, size, size, borderType=cv2.BORDER_CONSTANT, value=0.0)
        mask_ = cv2.copyMakeBorder(mask, size, size, size, size, borderType=cv2.BORDER_CONSTANT, value=0.0)
        return image_, mask_


class Crop(object):
    def __init__(self, min_size=0.5, min_ratio=0.5, max_ratio=2.0, p=0.25):
        self.min_size = min_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.p = p

    def __call__(self, image, mask):
        if np.random.uniform(0.0, 1.0) > self.p:
            return image, mask
        h, w, _ = image.shape
        aspect_ratio = np.random.uniform(self.min_ratio, self.max_ratio)  # = w / h
        if aspect_ratio > 1:
            w_ = int(np.random.uniform(self.min_size, 1.0) * w)
            h_ = int(w / aspect_ratio)
        else:
            h_ = int(np.random.uniform(self.min_size, 1.0) * h)
            w_ = int(h * aspect_ratio)

        x = np.random.randint(0, max(1, w - w_))
        y = np.random.randint(0, max(1, h - h_))
        crop_image = image[y: y + h_, x: x + w_, :]
        crop_mask = mask[y: y + h_, x: x + w_]
        return crop_image, crop_mask


class Resize(object):
    def __init__(self, size, keep_aspect=False):
        self.size = size
        self.keep_aspect = keep_aspect

    def __call__(self, image, mask):
        image_, mask_ = image.copy(), mask.copy()
        if self.keep_aspect:
            # padding step
            h, w = image.shape[:2]
            k = min(self.size[0] / w, self.size[1] / h)
            h_ = int(h * k)
            w_ = int(w * k)

            interpolation = cv2.INTER_AREA if k <= 1 else cv2.INTER_LINEAR
            image_ = cv2.resize(image_, None, fx=k, fy=k, interpolation=interpolation)
            mask_ = cv2.resize(mask_, None, fx=k, fy=k, interpolation=interpolation)

            dh = max(0, (self.size[1] - h_) // 2)
            dw = max(0, (self.size[0] - w_) // 2)
            image_ = cv2.copyMakeBorder(image_, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=0.0)
            mask_ = cv2.copyMakeBorder(mask_, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=0.0)
        if image_.shape[0] != self.size[1] or image_.shape[1] != self.size[0]:
            image_ = cv2.resize(image_, self.size)
            mask_ = cv2.resize(mask_, self.size)
        return image_, mask_


class Flip(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, mask):
        if np.random.uniform() > self.p:
            return image, mask
        return cv2.flip(image, 1), cv2.flip(mask, 1)

class ScaleToZeroOne(object):
    def __call__(self, image, mask):
        return np.float32(image / 255.), np.float32(mask / 255.)
    
    
class RandomRotate(object):
    def __init__(self, p=0.2, limit=90):
        self.p = p
        self.limit = limit
        self.augmenter = albu.Rotate(p=self.p, limit=self.limit)
        
    def __call__(self, image, mask):
        augmented = self.augmenter(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        for i in range(len(self.mean)):
            image[..., i] = (image[..., i] - self.mean[i]) / self.std[i]
#             mask[..., i] = (mask[..., i] - self.mean[i]) / self.std[i]
        return image, mask