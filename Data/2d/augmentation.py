from __future__ import division
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from torchvision.transforms import functional as F
from PIL import Image
import math

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, masks=None):
        for t in self.transforms:
            img, boxes, labels, masks = t(img, boxes, labels, masks)
        return img, boxes, labels,masks


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None, masks=None):
        return self.lambd(img, boxes, labels,masks)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, masks=None):
        if type(image)==np.ndarray:
            return image.astype(np.float32), boxes, labels, masks
        else:
            return np.array(image,dtype=np.float32), boxes, labels, np.array(masks,dtype=np.float32)

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, masks=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels, masks


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None, masks=None):
        if len(image.shape)==3:
            height, width, channels = image.shape
        else:
            height, width= image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels,masks


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, masks=None):
        if len(image.shape)==3:
            height, width, channels = image.shape
        else:
            height, width= image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels,masks


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None, masks=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        masks = cv2.resize(masks, (self.size,
                                 self.size))
        
        #masks = np.array(masks>0,dtype=np.int)
        return image, boxes, labels, masks


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            image[:, :, 0] *= random.uniform(self.lower, self.upper)
            image[:, :, 2] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels,masks



class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, masks


class RandomAddNoise(object):
    def __init__(self,delta=30.0):
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            image += random.uniform(-self.delta, self.delta,image.shape)
        return image, boxes, labels,masks
    
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels,masks


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None,masks=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels,masks


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels,masks


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels,masks



class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None, masks=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels,masks.cpu().numpy().astype(np.int)


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None,masks=None):
        if len(cvimage.shape)==3:
            if cvimage.max()>10:
                return torch.from_numpy(cvimage.astype(np.float32))[:,:,:3].permute(2, 0, 1)/255.0, torch.from_numpy(boxes).float(), torch.from_numpy(labels).long(),torch.from_numpy(masks)/255.0
            else:
                return torch.from_numpy(cvimage.astype(np.float32))[:,:,:3].permute(2, 0, 1), torch.from_numpy(boxes).float(), torch.from_numpy(labels).long(),torch.from_numpy(masks)
        else:
            #print(cvimage.shape,cvimage.min())
            if cvimage.max()>10:
                return (torch.from_numpy(cvimage.copy())[:,:,]/255.0).unsqueeze(0), torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),torch.from_numpy(masks.copy())/255.0
                
            else:
                return torch.from_numpy(cvimage.copy())[:,:,].unsqueeze(0), torch.from_numpy(boxes.copy()).float(), torch.from_numpy(labels.copy()).long(),torch.from_numpy(masks.copy())
            

class To255(object):
    def __call__(self, cvimage, boxes=None, labels=None,masks=None):
        if cvimage.max()<10:
            return cvimage*255.0,boxes,labels,masks
        else:
            return cvimage,boxes,labels,masks
        
        
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self,sample_options = None):
        if sample_options ==None:
            self.sample_options = (
                # using entire original input image
                None,
                # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
                (0.1, None),
                (0.3, None),
                (0.7, None),
                (0.9, None),
                # randomly sample a patch
                (None, None),)
        else:
            self.sample_optioins = sample_options
    def __call__(self, image, boxes=None, labels=None,masks=None):
        if len(image.shape)==3:
            height, width, channels = image.shape
        else:
            height, width= image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                current_masks = masks
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]
                current_masks = current_masks[rect[1]:rect[3], rect[0]:rect[2]]                                              

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels, current_masks

            
class RandomResizedCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, size, scale=(0.04, 0.25), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        """ Change this according to the original image sie """
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w < img.shape[0] and h < img.shape[1]:
                i = random.randint(0, img.shape[1] - h)
                j = random.randint(0, img.shape[0] - w)
                return i, j, h, w

        # Fallback

        w = min(img.shape[0], img.shape[1])
        i = (img.shape[1] - w) // 2
        j = (img.shape[0] - w) // 2
        return i, j, w, w
        
    def __call__(self, image, boxes=None, labels=None,masks=None):
        while True:
            # max trails (50)
            for _ in range(50):
                current_image = image
                current_masks = masks
                left,top,w,h = self.get_params(current_image,self.scale,self.ratio)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                
                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                #overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                #if overlap.min() < min_iou and max_iou < overlap.max():
                    #continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2]]
                current_masks = current_masks[rect[1]:rect[3], rect[0]:rect[2]]      

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                #if not mask.any():
                    #continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels,current_masks


class Expand(object):
    def __init__(self, mean,ratio = (1,1.1)):
        self.mean = mean
        self.ratio = ratio

    def __call__(self, image, boxes=None, labels=None, masks=None):
        if image.max() >10:
            mean = np.array(self.mean)*255.0
        if random.randint(2):
            return image, boxes, labels,masks
        
        if len(image.shape)==3:
            height, width, depth= image.shape
        else:
            height, width= image.shape
            depth = 0
        
        if self.ratio == None:
            ratio = random.uniform(1, 2)
        else:
            ratio = random.uniform(self.ratio[0],self.ratio[1])
            
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        
        if depth==0:
            expand_image = np.zeros((int(height*ratio), int(width*ratio)),dtype=image.dtype)
            expand_image[:, :] = mean
        else:
            expand_image = np.zeros((int(height*ratio), int(width*ratio), depth),dtype=image.dtype)
            expand_image[:, :, :] = mean
        
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        expand_masks = np.zeros((int(height*ratio), int(width*ratio)), dtype=masks.dtype)
        expand_masks[int(top):int(top + height),
                        int(left):int(left + width)] = masks

        masks = expand_masks

        #boxes = boxes.copy()
        #boxes[:, :2] += (int(left), int(top))
        #boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels,masks


class RandomMirror(object):
    def __call__(self, image, boxes=None, labels =None,masks=None):
        if len(image.shape)==3:
            _, width, _ = image.shape
        else:
            _, width = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            masks = masks[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels,masks


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self,  image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self,delta=18.0,con_lower=0.5, con_upper=1.5,sat_lower=0.5, sat_upper=1.5):
        self.pd = [
            RandomContrast(lower=con_lower, upper=con_upper),
            ConvertColor(transform='HSV'),
            RandomSaturation(lower=sat_lower, upper=sat_upper),
            RandomHue(delta=delta),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(lower=con_lower, upper=con_upper)
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()


    def __call__(self, image, boxes, labels,masks):
        im = image.copy()
        im, boxes, labels,masks = self.rand_brightness(im, boxes, labels,masks)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1])
        im, boxes, labels,masks= distort(im, boxes, labels, masks)
        return self.rand_light_noise(im, boxes, labels, masks)

class PhotometricDistort_grey(object):
    def __init__(self,delta=18.0,con_lower=0.5, con_upper=1.5):
        self.rand_contrast = RandomContrast(lower=con_lower, upper=con_upper)
        self.rand_brightness = RandomBrightness()
        self.rand_noise = RandomAddNoise(delta)


    def __call__(self, image, boxes, labels,masks):
        im = image.copy()
        im, boxes, labels,masks = self.rand_brightness(im, boxes, labels,masks)
        
        #distort = Compose(self.pd[0])
        im, boxes, labels,masks= self.rand_contrast(im, boxes, labels, masks)
        im, boxes, labels,masks= self.rand_noise(im, boxes, labels, masks)
        #return self.rand_light_noise(im, boxes, labels, masks)
        return im, boxes, labels, masks



class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes=None, labels=None ,masks = None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if image.shape[0]==1:
            #print(self.mean)
            return (image-self.mean)/self.std,boxes,labels,masks

        return F.normalize(image, self.mean, self.std),boxes,labels,F.normalize(mask, self.mean, self.std)

