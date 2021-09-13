import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import cv2

from detect import YOLODetector, VOC_CLASS_BGR


def compute_average_precision(recall, precision):
    """ Compute AP for one class.
    Args:
        recall: (numpy array) recall values of precision-recall curve.
        precision: (numpy array) precision values of precision-recall curve.
    Returns:
        (float) average precision (AP) for the class.
    """
    # AP (AUC of precision-recall curve) computation using all points interpolation.
    # For mAP computation, you can find a great explaination below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    ap = 0.0  # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap


def evaluate(preds, targets, class_names, threshold=0.5):
    """ Compute mAP metric.
    Args:
        preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.
        class_names: (list) list of class names.
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
    """
    # For mAP computation, you can find a great explanation below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    aps = []  # list of average precisions (APs) for each class.

    for class_name in class_names:
        class_preds = preds[class_name]  # all predicted objects for this class.

        if len(class_preds) == 0:
            ap = 0.0  # if no box detected, assign 0 for AP of this class.
            print('class {} AP {}'.format(class_name, ap))
            aps.append(ap)
            break

        image_fnames = [pred[0] for pred in class_preds]
        probs = [pred[1] for pred in class_preds]
        boxes = [pred[2:] for pred in class_preds]

        # Sort lists by probs.
        sorted_idxs = np.argsort(probs)[::-1]
        image_fnames = [image_fnames[i] for i in sorted_idxs]
        boxes = [boxes[i] for i in sorted_idxs]

        # Compute total number of ground-truth boxes. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, class_name_gt) in targets:
            if class_name_gt == class_name:
                num_gt_boxes += len(targets[filename_gt, class_name_gt])

        # Go through sorted lists, classifying each detection into TP or FP.
        num_detections = len(boxes)
        tp = np.zeros(num_detections)  # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (filename, box) in enumerate(zip(image_fnames, boxes)):

            if (filename, class_name) in targets:
                boxes_gt = targets[(filename, class_name)]
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if iou >= threshold:
                        tp[det_idx] = 1.0
                        fp[det_idx] = 0.0

                        boxes_gt.remove(box_gt)  # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[(filename, class_name)]  # remove empty element from the dictionary.

                        break

            else:
                pass  # this detection is FP.

        # Compute AP from `tp` and `fp`.
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        eps = np.finfo(np.float64).eps
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        recall = tp_cumsum / float(num_gt_boxes)

        ap = compute_average_precision(recall, precision)

        print(f'{class_name}'.ljust(25, ' '), f'{ap:.2f}')
        aps.append(ap)

    return aps


if __name__ == '__main__':

    VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    # Path to the yolo weight to test.
    model_path = 'weights/best.pth'

    # Path to image dir.
    image_dir = '../../Datasets/VOC/IMAGES'

    voc_class_names = list(VOC_CLASS_BGR.keys())

    targets = defaultdict(list)
    preds = defaultdict(list)

    print('DATA PREPARING...')
    with open('../../Datasets/VOC/test.txt') as f:
        lines = f.readlines()
    image_list = []
    for line in lines:
        line = line.strip()
        image_name = f'{line}.jpg'
        image_list.append(image_name)

        with open(f'../../Datasets/VOC/LABELS/{line}.txt') as f:
            objects = f.readlines()

        for object in objects:
            c, x1, y1, x2, y2 = map(int, object.rstrip().split())
            class_name = voc_class_names[c]
            targets[(image_name, class_name)].append([x1, y1, x2, y2])
    print('DONE.\n')
    print('START TESTING...')

    # Load YOLO model.
    yolo = YOLODetector(model_path, conf_thresh=-1.0, prob_thresh=-1.0, nms_thresh=0.5)

    # Detect objects with the model.
    progress_bar = tqdm(image_list, total=len(image_list))
    for filename in progress_bar:
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        boxes, class_names, probs = yolo.detect(image)
        for box, class_name, prob in zip(boxes, class_names, probs):
            x1y1, x2y2 = box
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            preds[class_name].append([filename, prob, x1, y1, x2, y2])

    print('Evaluate the detection result...')

    aps = evaluate(preds, targets, class_names=voc_class_names)

    # Compute mAP by averaging APs for all classes.
    print('mAP {}'.format(np.mean(aps)))
