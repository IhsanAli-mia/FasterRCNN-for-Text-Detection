import torch
import numpy as np
import cv2
import torchvision
import argparse
import random
import os
import yaml
from tqdm import tqdm
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt

import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score], ...],
    #       'car' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2], ...],
    #       'car' : [[x1, y1, x2, y2], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    # gt_labels.remove('background')
    all_aps = {}
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps

def get_rotated_iou(det, gt, det_angle, gt_angle):
    """
    Calculate IoU between two rotated bounding boxes
    
    Args:
        det: [x1, y1, x2, y2] of detected box (min/max coordinates)
        gt: [x1, y1, x2, y2] of ground truth box (min/max coordinates)
        det_angle: angle in radians for detected box
        gt_angle: angle in radians for ground truth box
        
    Returns:
        IoU value between 0 and 1
    """
    # Convert from min/max format to rotated rectangle format
    det_cx = (det[0] + det[2]) / 2
    det_cy = (det[1] + det[3]) / 2
    det_width = det[2] - det[0]
    det_height = det[3] - det[1]
    
    gt_cx = (gt[0] + gt[2]) / 2
    gt_cy = (gt[1] + gt[3]) / 2
    gt_width = gt[2] - gt[0]
    gt_height = gt[3] - gt[1]
    
    # print(type(det_angle))
    # print(type(gt_angle))
    
    # Create rotated rectangles
    det_rect = ((det_cx, det_cy), (det_width, det_height), det_angle)
    gt_rect = ((gt_cx, gt_cy), (gt_width, gt_height), gt_angle)
    
    # Convert to point representation for cv2.rotatedRectangleIntersection
    det_points = cv2.boxPoints(det_rect)
    gt_points = cv2.boxPoints(gt_rect)
    
    
    # print(det_rect)
    # print(gt_rect)
    
    
    # return _,_
    # Calculate intersection
    intersection_result = cv2.rotatedRectangleIntersection(det_rect, gt_rect)
    
    if intersection_result[0] == 0:  # No intersection
        return 0.0
    
    # Calculate areas
    det_area = det_width * det_height
    gt_area = gt_width * gt_height
    
    # Calculate intersection area
    intersection_points = intersection_result[1]
    if len(intersection_points) < 3:  # Need at least 3 points to form a polygon
        return 0.0
    
    intersection_area = cv2.contourArea(intersection_points)
    
    # Calculate union area
    union_area = det_area + gt_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / (union_area + 1e-6)
    
    return iou

def angle_similarity(angle1, angle2):
    """
    Calculate similarity between two angles, handling periodicity
    
    Args:
        angle1, angle2: angles in radians
        
    Returns:
        Similarity score between 0 and 1 (1 means identical angles)
    """
    # Normalize angles to [0, π] since orientations that differ by π are equivalent
    # for rectangular boxes
    angle1 = angle1 % np.pi
    angle2 = angle2 % np.pi
    
    # Calculate the minimum angular difference
    diff = min(abs(angle1 - angle2), np.pi - abs(angle1 - angle2))
    
    # Convert to similarity score (1 when diff=0, 0 when diff=π/2)
    similarity = max(0, 1 - 2 * diff / np.pi)
    
    return similarity

def get_iou_with_angle(det, gt, det_angle, gt_angle, angle_weight=0.3):
    """
    Calculate IoU with angle penalty
    
    Args:
        det, gt: Bounding boxes [x1, y1, x2, y2]
        det_angle, gt_angle: Angles in radians
        angle_weight: Weight of angle similarity in final score (0-1)
        
    Returns:
        Modified IoU score considering angle
    """
    # Get standard IoU
    iou = get_rotated_iou(det, gt, det_angle, gt_angle)
    
    # Get angle similarity
    angle_sim = angle_similarity(det_angle, gt_angle)
    
    # Combine IoU and angle similarity
    # When angle_weight=0, it's just standard IoU
    # When angle_weight=1, it's just angle similarity
    combined_score = (1 - angle_weight) * iou + angle_weight * angle_sim * iou
    
    return combined_score

def compute_map_with_angle(det_boxes, gt_boxes, det_angles, gt_angles, iou_threshold=0.5, angle_weight=0.3, method='area'):
    """
    Compute mAP with angle consideration
    
    Args:
        det_boxes: List of dictionaries containing detection boxes by class
        gt_boxes: List of dictionaries containing ground truth boxes by class
        det_angles: List of dictionaries containing detection angles by class
        gt_angles: List of dictionaries containing ground truth angles by class
        iou_threshold: IoU threshold for matching
        angle_weight: Weight of angle in IoU calculation (0-1)
        method: 'area' or 'interp' for AP calculation method
        
    Returns:
        mean_ap: Mean average precision
        all_aps: Dictionary of AP values by class
    """
    # Get all unique class labels
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    
    all_aps = {}
    aps = []
    
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class with image index
        cls_dets = []
        for im_idx, im_dets in enumerate(det_boxes):
            if label in im_dets:
                for det_idx, det in enumerate(im_dets[label]):
                    # Store (image_index, detection, angle)
                    det_angle = det_angles[im_idx][label][det_idx]
                    cls_dets.append((im_idx, det, det_angle))
        
        # Sort by confidence score (last element of det)
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])
        
        # For tracking which gt boxes have been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes if label in im_gts]
        
        # Total number of ground truth boxes for this class
        num_gts = sum([len(im_gts[label]) if label in im_gts else 0 for im_gts in gt_boxes])
        
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)
        
        # For each detection
        for det_idx, (im_idx, det_pred, det_angle) in enumerate(cls_dets):
            if label not in gt_boxes[im_idx]:
                fp[det_idx] = 1
                continue
                
            # Get ground truth boxes for this image and label
            im_gts = gt_boxes[im_idx][label]
            im_gt_angles = gt_angles[im_idx][label]
            
            max_iou_found = -1
            max_iou_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_angle = im_gt_angles[gt_box_idx]
                
                # Calculate IoU with angle consideration
                iou_with_angle = get_iou_with_angle(
                    det_pred[:-1],  # Remove score
                    gt_box,
                    det_angle,
                    gt_angle,
                    angle_weight
                )
                
                if iou_with_angle > max_iou_found:
                    max_iou_found = iou_with_angle
                    max_iou_gt_idx = gt_box_idx
            
            # Check if match is good enough and not already matched
            if max_iou_found < iou_threshold or (max_iou_gt_idx >= 0 and gt_matched[im_idx][max_iou_gt_idx]):
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # Mark this ground truth as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        
        # Compute cumulative TP and FP
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        # Calculate precision and recall
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)
        
        # Calculate AP based on method
        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            
            # Compute precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
                
            # Find points where recall changes
            i = np.where(recalls[1:] != recalls[:-1])[0]
            
            # Calculate AP as area under precision-recall curve
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
            
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                prec_interp_pt = precisions[recalls >= interp_pt]
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
            
        else:
            raise ValueError('Method can only be area or interp')
            
        # Store AP for this class
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
            
    # Calculate mean AP across all classes
    mean_ap = sum(aps) / len(aps) if len(aps) > 0 else 0.0
    
    return mean_ap, all_aps

def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    angle_step_size = dataset_config['angle_step_size']
    prediction_method = train_config['angle_prediction_method']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    st = SceneTextDataset('val', root_dir=dataset_config['root_dir'],angle_step_size=dataset_config['angle_step_size'],prediction_method=train_config['angle_prediction_method'])
    test_dataset = DataLoader(st, batch_size=1, shuffle=False)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                            min_size=600,
                                                            max_size=1000,
                                                            box_score_thresh=0.7,
                                                            angle_step_size=dataset_config['angle_step_size'],
                                                            prediction_method=train_config['angle_prediction_method']
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'])

    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    
    model_name = 'tv_frcnn_r50fpn_'
    
    if prediction_method == 'classification':
        model_name = 'classification_oriented_' + model_name
    
    faster_rcnn_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                                model_name + train_config['ckpt_name']),
                                                    map_location=device))

    return faster_rcnn_model, st, test_dataset, [angle_step_size,prediction_method]

def visualise(args):
    visual_dir = 'visualization_oriented'
    if not os.path.exists(visual_dir):
        os.mkdir(visual_dir)
        
    _, voc, _ = load_model_and_dataset(args)
    
    for sample_count in tqdm(range(10)):
        random_idx = random.randint(0,len(voc))
        im, target, fname = voc[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        # Saving images with ground truth boxes
        for idx, [box, theta] in enumerate(zip(target['bboxes'],target['theta'])):
            x1, y1, x2, y2= box.detach().cpu().numpy()
            theta = theta.detach().cpu().numpy()
            x1, y1, x2, y2, theta = int(x1), int(y1), int(x2), int(y2), int(theta)

            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])

            text = voc.idx2label[target['labels'][idx].detach().cpu().item()]
            text_with_angle = f"{text} ({theta} deg)"  # Adding angle to the text

            text_size, _ = cv2.getTextSize(text_with_angle, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            
            cv2.putText(gt_im, text=text_with_angle, 
                        org=(x1 + 5, y1 + 15), 
                        thickness=1, 
                        fontScale=1, 
                        color=[0, 0, 0], 
                        fontFace=cv2.FONT_HERSHEY_PLAIN)

            cv2.putText(gt_im_copy, text=text_with_angle, 
                        org=(x1 + 5, y1 + 15), 
                        thickness=1, 
                        fontScale=1, 
                        color=[0, 0, 0], 
                        fontFace=cv2.FONT_HERSHEY_PLAIN)

        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('{}/gt_{}.png'.format(visual_dir, sample_count), gt_im)

        
        
def infer(args):
    output_dir = 'oriented_samples_tv_r50fpn'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    faster_rcnn_model, voc, test_dataset, angle_parameters = load_model_and_dataset(args)
    
    angle_step_size,angle_prediction = angle_parameters
    
    visual_dir = 'visualization'
    if not os.path.exists(visual_dir):
        os.mkdir(visual_dir)
    
    for sample_count in tqdm(range(10)):
        random_idx = random.randint(0,len(voc))
        im, target, fname = voc[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        cv2.imwrite('{}/visualization_{}.png'.format(visual_dir, sample_count), gt_im)

        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            theta = target['theta'][idx].detach().cpu()
            if angle_prediction == 'classification':
                theta = torch.argmax(theta, dim=-1) * (180 / (len(theta)-1))
            text = f"{theta.item()} deg : {voc.idx2label[target['labels'][idx].detach().cpu().item()]}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=voc.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('{}/output_frcnn_gt_{}.png'.format(output_dir, sample_count), gt_im)

        # Getting predictions from trained model
        frcnn_output = faster_rcnn_model(im, None)[0]
        # print(frcnn_output.keys())
        # frcnn_output = frcnn_output[0]
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        thetas = frcnn_output['angles']
        
        # print(boxes)
        # print(thetas)
        
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            theta = thetas[idx].detach().cpu()
            if angle_prediction == 'classification':
                theta = torch.argmax(theta, dim=-1) * (180 / (len(theta)-1))
            else:
                theta = theta.item()
            text = '{} deg: {:.2f}'.format(theta,
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite('{}/output_frcnn_{}.jpg'.format(output_dir, sample_count), im)


def evaluate_map(args):
    faster_rcnn_model, voc, test_dataset,angle_parameters = load_model_and_dataset(args)
    
    angle_step_size,angle_prediction = angle_parameters
    
    mean_aps = []
    
    # print(len(voc.label2idx))
    # print(voc.label2idx)
    
    for i in range(10,15):
        
        checkpoint_path = f'./checkpoints/oriented_checkpoint_epoch_{i}.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        faster_rcnn_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    
        gts = []
        preds = []
        det_angles = []
        gt_angles = []
        
        for im, target, _ in tqdm(test_dataset):
            # im_name = fname
            # print(target)
            im = im.float().to(device)
            target_boxes = target['bboxes'].float().to(device)[0]
            target_labels = target['labels'].long().to(device)[0]
            # print(target_boxes)
            # print(target_labels)
            theta = target['theta'].float().to(device)[0]
            
            # print(target_boxes.size())
            # print(target_labels.size())
            # print(theta.size())
            
            frcnn_output = faster_rcnn_model(im, None)[0]
            

            boxes = frcnn_output['boxes']
            labels = frcnn_output['labels']
            scores = frcnn_output['scores']
            angles = frcnn_output['angles']
            
            det_angle_dict = {label: [] for label in voc.idx2label.values()}
            gt_angle_dict = {label: [] for label in voc.idx2label.values()}
        
            pred_boxes = {}
            gt_boxes = {}
            for label_name in voc.label2idx:
                pred_boxes[label_name] = []
                gt_boxes[label_name] = []

            # print(pred_boxes)

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                label = labels[idx].detach().cpu().item()
                score = scores[idx].detach().cpu().item()
                angle = angles[idx].detach().cpu().item()
                label_name = voc.idx2label[label]

                # Store box in pred_boxes (existing code)
                pred_boxes[label_name].append([x1, y1, x2, y2, score])

                # Store angle in det_angle_dict (new code)
                det_angle_dict[label_name].append(angle)
                
            # print(pred_boxes)
            # print(len(pred_boxes['text']))
            for idx, box in enumerate(target_boxes):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                label = target_labels[idx].detach().cpu().item()
                label_name = voc.idx2label[label]
                angle = theta[idx].detach().cpu().item()
            
            # Store box in gt_boxes (existing code)
                gt_boxes[label_name].append([x1, y1, x2, y2])

                # Store angle in gt_angle_dict (new code)
                gt_angle_dict[label_name].append(angle)



            gts.append(gt_boxes)
            preds.append(pred_boxes)
            det_angles.append(det_angle_dict)
            gt_angles.append(gt_angle_dict)
            # break

        print(voc.idx2label)
        mean_ap, all_aps = compute_map_with_angle(preds, gts,  det_angles, gt_angles, method='area')
        print('Class Wise Average Precisions')
        for idx in range(1,2):
            print('AP for class {} = {:.4f}'.format(voc.idx2label[idx], all_aps[voc.idx2label[idx]]))
        print('Mean Average Precision : {:.4f}'.format(mean_ap))
        
        mean_aps.append(mean_ap)
        
    # epochs = list(range(1, 16))  # 15 epochs

    # plt.figure(figsize=(8, 5))
    # plt.plot(epochs, mean_aps, marker='o', linestyle='-', color='b', label='Mean AP')
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Average Precision (mAP)')
    # plt.title('mAP Progress Over 15 Epochs')
    # plt.xticks(epochs)  # Ensure all epochs are labeled
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend()
    # plt.show()
    
    
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inference using torchvision code faster rcnn')
    parser.add_argument('--config', dest='config_path',
                        default='config/st.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=False, type=bool)
    parser.add_argument('--visualise', dest='visualise',
                        default=False, type=bool)
    args = parser.parse_args()
    
    if args.infer_samples:
        infer(args)
    else:
        print('Not Inferring for samples as `infer_samples` argument is False')

    if args.evaluate:
        evaluate_map(args)
    else:
        print('Not Evaluating as `evaluate` argument is False')
        
    if args.visualise:
        visualise(args)
    else:
        print('Not Visualising as `visualise` argument is False')