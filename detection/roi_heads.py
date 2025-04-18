from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align

from . import _utils as det_utils


def fastrcnn_loss(class_logits, box_regression, angle_prediction, labels, regression_targets,angle_targets,angle_bins=18,angle_method='regression'):
    # type: (Tensor, Tensor, Tensor, List[Tensor], List[Tensor],List[Tensor],int,String) -> Tuple[Tensor, Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        angle_prediction (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
        angle_targets (Tensor)
        angle_bins (int)
        angle_method (String)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
        angle_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    # angle_targets = torch.cat(angle_targets, dim=0).view(-1, 1)  # Ensure shape [N, 1]

    # print(angle_prediction)
    # print(angle_prediction.size())

    # print(angle_targets)
    # print(angle_targets[0].size())

    # print(class_logits)
    
    # print(labels)
    
    # print(class_logits.size())
    
    # print(labels.size())
    
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    # if angle_method == 'classification':
    #     angle_prediction = angle_prediction.reshape(N, angle_bins)  # Reshape for classification
    # else:
    #     angle_prediction = angle_prediction.reshape(N, 1) 
    
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()
    
    # print(box_regression.shape)
    # print(regression_targets.shape)
    # print(angle_prediction.shape)
    # print(len(angle_targets))
    # print(angle_prediction.device)
    # print(angle_targets[0].device)
    
    # angle_targets[0].to(device='cuda')
    
    if angle_method == 'regression':
        angle_targets = torch.cat(angle_targets, dim=0).view(-1, 1)
        angle_loss = F.smooth_l1_loss(
            angle_prediction.view(-1, 1)[sampled_pos_inds_subset].to(device='cuda'),  # Ensure proper indexing
            angle_targets[sampled_pos_inds_subset].to(device='cuda')
        )
    elif angle_method == 'classification':
        angle_loss = F.cross_entropy(angle_prediction, angle_targets[0].to(device='cuda'))
    else:
        raise ValueError("angle_method must be either 'regression' or 'classification'")

    return classification_loss, box_loss,angle_loss

class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        angle_step_size=0,
        prediction_method='regression',
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.angle_step_size = angle_step_size
        self.prediction_method = prediction_method
        self.angle_bins = 180//angle_step_size if angle_step_size != 0 else 1
        if self.prediction_method == 'classification':
            self.angle_predictor = nn.Linear(self.box_predictor.cls_score.in_features, self.angle_bins + 1)
        else:
            self.angle_predictor = nn.Linear(self.box_predictor.cls_score.in_features, 1)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")

    def encode_angles(self, reference_angles: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        angles_per_image = [len(a) for a in reference_angles]
        reference_angles = torch.cat(reference_angles, dim=0)
        proposals = torch.cat(proposals, dim=0)
        angle_targets = reference_angles  # No transformation needed for angles
        return angle_targets.split(angles_per_image, 0)
    
    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_angles = [t["theta"].to(dtype) for t in targets]  # Extract ground truth angles

        proposals = self.add_gt_proposals(proposals, gt_boxes)

        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        matched_gt_angles = []
        num_images = len(proposals)
        
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            gt_angles_in_image = gt_angles[img_id]

            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
                gt_angles_in_image = torch.zeros((1,), dtype=dtype, device=device)

            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            matched_gt_angles.append(gt_angles_in_image[matched_idxs[img_id]])  # Assign correct angles

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        # angle_targets = self.encode_angles(matched_gt_angles,proposals)
        
        return proposals, matched_idxs, labels, regression_targets, matched_gt_angles
    
    def postprocess_detections(self, class_logits, box_regression, angle_logits, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        if self.prediction_method == 'classification':
            pred_angles = F.softmax(angle_logits,-1)
        else:
            pred_angles = angle_logits
        pred_angles_list = pred_angles.split(boxes_per_image,0)

        all_boxes, all_scores, all_labels, all_angles = [], [], [], []
        for boxes, scores, angles, image_shape in zip(pred_boxes_list, pred_scores_list, pred_angles_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device).view(1, -1).expand_as(scores)
            boxes, scores, labels = boxes[:, 1:], scores[:, 1:], labels[:, 1:]
            boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, angles = boxes[keep], scores[keep], labels[keep], angles[keep]

            
            angles = angles.squeeze(dim=-1)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_angles.append(angles)
        
        return all_boxes, all_scores, all_labels, all_angles

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets,angle_targets = self.select_training_samples(proposals, targets)
            # angle_targets = [t["theta"] for t in targets]
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
            angle_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        angle_logits = self.angle_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg, loss_angles = fastrcnn_loss(class_logits, box_regression, angle_logits, labels, regression_targets, angle_targets,angle_bins=self.angle_bins,angle_method=self.prediction_method)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_angles": loss_angles}
        else:
            boxes, scores, labels, angles = self.postprocess_detections(class_logits, box_regression,angle_logits, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "angles": angles[i],
                    }
                )

        return result, losses