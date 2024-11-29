import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import generalized_box_iou_loss
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead
class GIouRoIHeads(RoIHeads):
    def __init__(self,*args,**kwargs):
        super(GIouRoIHeads,self).__init__(*args,**kwargs)
    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        classification_loss = F.cross_entropy(class_logits, labels)
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        if sampled_pos_inds_subset.numel() == 0:
            # No positive samples, return zero box loss
            box_loss = torch.tensor(0.0, device=labels.device)
            return classification_loss, box_loss
        box_regression = box_regression[sampled_pos_inds_subset]
        regression_targets = regression_targets[sampled_pos_inds_subset]

        boxes_per_image = [len(sampled_pos_inds_subset)]
        pred_boxes = self.box_coder.decode(box_regression, [regression_targets])

        # Get the target boxes (regression targets are encoded deltas)
        device = regression_targets.device
        gt_boxes = self.box_coder.decode(torch.zeros_like(box_regression), [regression_targets])

        gio_loss=generalized_box_iou_loss(pred_boxes[0],gt_boxes[0],reduction='mean')
        return classification_loss,gio_loss

def get_object_detection_model_giou(num_classes):
    backbone=resnet_fpn_backbone('resnet50',pretrained=True)
    anchor_sizes = ((32,),(64,),(128,),( 256,), (512,))  
    aspect_ratios = ((0.5, 0.75, 1.0, 1.5, 2.0),(0.5, 0.75, 1.0, 1.5, 2.0),(0.5, 0.75, 1.0, 1.5, 2.0),(0.5, 0.75, 1.0, 1.5, 2.0),(0.5, 0.75, 1.0, 1.5, 2.0))


    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model=FasterRCNN(backbone,rpn_anchor_generator=anchor_generator,num_classes=2)
    model.roi_heads = GIouRoIHeads(
        box_roi_pool=model.roi_heads.box_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=model.roi_heads.box_predictor,
        fg_iou_thresh=model.roi_heads.proposal_matcher.high_threshold,
        bg_iou_thresh=model.roi_heads.proposal_matcher.low_threshold,
        batch_size_per_image=model.roi_heads.fg_bg_sampler.batch_size_per_image,
        positive_fraction=model.roi_heads.fg_bg_sampler.positive_fraction,
        bbox_reg_weights=None,
        score_thresh=model.roi_heads.score_thresh,
        nms_thresh=model.roi_heads.nms_thresh,
        detections_per_img=model.roi_heads.detections_per_img)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
def get_object_detection_model_restnet101(num_classes):
    anchor_sizes = ((128,),(256,),( 512,), (1024,),(2048,))  
    aspect_ratios = ((0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0))

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    backbone=resnet_fpn_backbone('resnet101',pretrained=True)
    model=FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,rpn_pre_nms_top_n_train=4000,rpn_post_nms_top_n_train=2000,rpn_pre_nms_top_n_test=2000,rpn_post_nms_top_n_test=1000,rpn_nms_thresh=0.85,rpn_fg_iou_thresh=0.8,rpn_bg_iou_thresh=0.5,box_fg_iou_thresh=0.7,box_bg_iou_thresh=0.5,box_batch_size_per_image=512,box_positive_fraction=0.25,)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
def get_object_detection_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.rpn.parameters():
        param.requires_grad = True

    for param in model.roi_heads.parameters():
        param.requires_grad = True
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
def get_object_detection_model_fl(num_classes):
    backbone=resnet_fpn_backbone('resnet50',pretrained=True)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    custom_model = FRCNN_FocalLoss(backbone, num_classes=2)
    #custom_model.to(device)
    in_features = custom_model.roi_heads.box_predictor.cls_score.in_features
    custom_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return custom_model
def get_object_detection_model_fcos(num_classes):
    model = fcos_resnet50_fpn(pretrained=True)
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = FCOSClassificationHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_anchors=num_anchors)
    return model
