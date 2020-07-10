import logging
import torch
import cv2
import numpy as np
def order_points(pts):
     
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, pts):
    
    rect = order_points(pts)
    
    tl, tr, br, bl = pts
    
    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_1), int(height_2))
    
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

def get_logger(filename: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def dice_coeff(input, target):
    smooth = 1.

    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()

    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(input, target):
    return - torch.log(dice_coeff(input, target))


def get_boxes_from_mask(mask, margin, clip=False):
    """
    Detect connected components on mask, calculate their bounding boxes (with margin) and return them (normalized).
    If clip is True, cutoff the values to (0.0, 1.0).
    :return np.ndarray boxes shaped (N, 4)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for j in range(1, num_labels):  # j = 0 == background component
        x, y, w, h = stats[j][:4]
        x1 = int(x - margin * w)
        y1 = int(y - margin * h)
        x2 = int(x + w + margin * w)
        y2 = int(y + h + margin * h)
        box = np.asarray([x1, y1, x2, y2])
        boxes.append(box)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype(np.float)
    boxes[:, [0, 2]] /= mask.shape[1]
    boxes[:, [1, 3]] /= mask.shape[0]
    if clip:
        boxes = boxes.clip(0.0, 1.0)
    return boxes

def resize(image, size, keep_aspect=False):
    image_= image.copy()
    if keep_aspect:
        # padding step
        h, w = image.shape[:2]
        k = min(size[0] / w, size[1] / h)
        h_ = int(h * k)
        w_ = int(w * k)
        interpolation = cv2.INTER_AREA if k <= 1 else cv2.INTER_LINEAR
        image_ = cv2.resize(image_, None, fx=k, fy=k, interpolation=interpolation)
        dh = max(0, (size[1] - h_) // 2)
        dw = max(0, (size[0] - w_) // 2)
        image_ = cv2.copyMakeBorder(image_, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=0.0)
    if image_.shape[0] != size[1] or image_.shape[1] != size[0]:
        image_ = cv2.resize(image_, size)
    return image_, k, dw, dh

def normalize(image, mean, std):
    for i in range(len(mean)):
        image[..., i] = (image[..., i] - mean[i]) / std[i]
    return image

def prepare_for_inference(image, fit_size, mean=None, std=None):
    """
    Scale proportionally image into fit_size and pad with zeroes to fit_size
    :return: np.ndarray image_padded shaped (*fit_size, 3), float k (scaling coef), float dw (x pad), dh (y pad)
    """
    # pretty much the same code as detection.transforms.Resize
    # resizing
    image_padded, k, dw, dh = resize(image, fit_size, keep_aspect=True)
    # normalizing
    if mean and std:
        image_padded = normalize(image_padded, mean, std)
    
    return image_padded, k, dw, dh

def _get_boxes_from_mask(mask, dw, dh, k):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    resized_contours = [resize_box_back(contour[:, 0], dw, dh, k) for contour in contours]
    boxes = [_contour2box(contour) for contour in resized_contours]
    return boxes

def _contour2box(contour):
    """
    convert contour to box

    :param contour: np.array[Nx2]: open cv contour
    :param angle: bool: if True the function return rotating boxes False - function return ordinary boxes
    :param output_format: str: output boxes format
    :return: np.array: box
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = convert_boxes([box])[0]
    return box

def resize_box_back(box, dw, dh, k):
    box_copy = box.copy()
    box_copy[:, 0] -= dw
    box_copy[:, 1] -= dh
    box_copy = box_copy / k
    return box_copy.astype(int)

def convert_boxes(boxes):
    new_boxes = []
    for box in boxes:
#         points = sorted(box, key=lambda point: point[1])
#         up_points = sorted(points[:2], key=lambda point: point[0])
#         bottom_points = sorted(points[2:], key=lambda point: point[0])
# #         if up_points[1][1] > bottom_points[0][1]:
            
#         new_box = [up_points[0], up_points[1], bottom_points[1], bottom_points[0]]
        new_box = order_points(box)
        new_boxes.append(new_box)
    return np.array(new_boxes)

def get_distance(point_1, point_2):
    """
    calculate euclidean distance between points

    :param point_1: (float, float): point coordinates (x, y)
    :param point_2: (float, float): point coordinates (x, y)
    :return: float: distance between points
    """
    result = ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5
    return result


def crop_bounding_box(image, bbox, flags=cv2.INTER_LINEAR):
    """
    crop sub-image by bounding box

    :param image: np.array: image
    :param bbox: array-like: box in 4_points format,  means 1----2 format, where each point is (x, y)
                                                            |    |
                                                            4----3
    :param flags: cv2.flag: flag for warpPerspective transform, use cv2.INTER_LINEAR for image and cv2.INTER_NEAREST
                            for masks
    :return: np.array: cropped image
    """
    box_points = bbox.astype(np.float32)

    result_width = int(max(get_distance(bbox[0], bbox[1]),
                           get_distance(bbox[2], bbox[3])))
    result_height = int(max(get_distance(bbox[0], bbox[3]),
                            get_distance(bbox[1], bbox[2])))
    if result_width > result_height:
        
        expected_box_points = np.array([[0, 0],
                                    [result_width, 0],
                                    [result_width, result_height],
                                    [0, result_height]], dtype=np.float32)
    else:
        expected_box_points = np.array([[0, 0],
                                    [result_width, 0],
                                    [result_width, result_height],
                                    [0, result_height]], dtype=np.float32)
    perspective_operator = cv2.getPerspectiveTransform(box_points, expected_box_points)

    cropped_image = cv2.warpPerspective(src=image,
                                        M=perspective_operator,
                                        dsize=(result_width, result_height),
                                        flags=flags)
    return cropped_image