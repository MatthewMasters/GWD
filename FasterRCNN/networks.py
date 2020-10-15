import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.jit.annotations import Tuple, List
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads as RoIHeadsOld
from torchvision.ops import boxes as box_ops
from torch.jit.annotations import Optional, List, Dict, Tuple
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d
from collections import OrderedDict


def get_resnet(freeze_backbone):
    backbone = resnet.__dict__['resnet50'](pretrained=True, norm_layer=FrozenBatchNorm2d)
    if freeze_backbone:
        for name, parameter in backbone.named_parameters():
            # if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)
    return backbone

class MyFasterRCNN(nn.Module):
    def __init__(self, use_focal_loss=False, freeze_backbone=False, mean='imagenet', image_size=1024, num_predictions=100):
        super(MyFasterRCNN, self).__init__()
        self.image_size = image_size

        # Normalize image
        if mean == 'imagenet':
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        elif mean == 'dataset':
            image_mean = [0.3151, 0.3179, 0.2150]
            image_std = [0.2456, 0.2387, 0.1944]
        self.transform = GeneralizedRCNNTransform(self.image_size, self.image_size, image_mean, image_std)

        self.freeze_backbone = freeze_backbone
        self.backbone = get_resnet(freeze_backbone=self.freeze_backbone)
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)

        # Feature Pyramid Network
        out_channels = 256
        in_channels_list = [256 * (2 ** i) for i in range(4)] # [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool())

        # Regional Proposal Network
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        head = RPNHead(out_channels, anchor_generator.num_anchors_per_location()[0])
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7)

        # RoI heads
        representation_size = 512
        box_roi_pool = MultiScaleRoIAlign(['0', '1', '2', '3'], 7, 2)
        box_head = TwoMLPHead(out_channels * box_roi_pool.output_size[0] ** 2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, num_classes=2)
        self.roi_heads = RoIHeadsOld(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=num_predictions)

    def forward(self, images, targets=None):
        images, targets = self.transform(images, targets)
        features = self.body(images.tensors)

        fpn_features = self.fpn(features)
        if isinstance(fpn_features, torch.Tensor): fpn_features = OrderedDict([('0', fpn_features)])
        proposals, proposal_losses = self.rpn(images, fpn_features, targets)
        detections, detector_losses = self.roi_heads(fpn_features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return features, detections, losses


class DetNetwork(nn.Module):
    def __init__(self, num_predictions=100):
        super(DetNetwork, self).__init__()
        # Feature Pyramid Network
        out_channels = 256
        in_channels_list = [256 * (2 ** i) for i in range(4)] # [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool())

        # Regional Proposal Network
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        head = RPNHead(out_channels, anchor_generator.num_anchors_per_location()[0])
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7)

        # RoI heads
        representation_size = 512
        box_roi_pool = MultiScaleRoIAlign(['0', '1', '2', '3'], 7, 2)
        box_head = TwoMLPHead(out_channels * box_roi_pool.output_size[0] ** 2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, num_classes=2)
        self.roi_heads = RoIHeadsOld(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=num_predictions)

    def forward(self, images, features, targets=None):
        fpn_features = self.fpn(features)
        if isinstance(fpn_features, torch.Tensor): fpn_features = OrderedDict([('0', fpn_features)])
        proposals, proposal_losses = self.rpn(images, fpn_features, targets)
        detections, detector_losses = self.roi_heads(fpn_features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return features, detections, losses


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp().detach()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, cls_loss_func=F.cross_entropy):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor])
    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor) (N, K)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = cls_loss_func(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def attention_crop_drop(attention_maps, input_image):
    # start = time.time()
    B, N, W, H = input_image.shape
    input_tensor = input_image
    batch_size, num_parts, height, width = attention_maps.shape

    # attention_maps = torch.nn.functional.interpolate(attention_maps.detach(),size=(W,H),mode='bilinear')
    attention_maps = attention_maps.detach()
    part_weights = F.avg_pool2d(attention_maps.detach(), (W, H)).reshape(batch_size, -1)
    part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
    part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu()
    part_weights = part_weights.numpy()
    # print(part_weights.shape)
    ret_imgs = []
    masks = []
    # print(part_weights[3])
    for i in range(batch_size):
        attention_map = attention_maps[i]
        part_weight = part_weights[i]
        selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        selected_index2 = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        ## create crop imgs
        mask = attention_map[selected_index, :, :]
        # mask = (mask-mask.min())/(mask.max()-mask.min())
        threshold = random.uniform(0.4, 0.6)
        # threshold = 0.5
        itemindex = np.where(mask.cpu() >= mask.cpu().max() * threshold)
        # print(itemindex.shape)
        # itemindex = torch.nonzero(mask >= threshold*mask.max())
        padding_h = max(int(0.1 * H), 1)
        padding_w = max(int(0.1 * W), 1)
        height_min = itemindex[0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[0].max() + padding_h
        width_min = itemindex[1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[1].max() + padding_w
        # print('numpy',height_min,height_max,width_min,width_max)
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        ret_imgs.append(out_img)

        ## create drop imgs
        mask2 = attention_map[selected_index2:selected_index2 + 1, :, :]
        threshold = random.uniform(0.2, 0.5)
        mask2 = (mask2 < threshold * mask2.max()).float()
        masks.append(mask2)
    # bboxes = np.asarray(bboxes, np.float32)
    crop_imgs = torch.stack(ret_imgs)
    masks = torch.stack(masks)
    drop_imgs = input_tensor * masks
    return (crop_imgs, drop_imgs)


def mask2bbox(attention_maps, input_image):
    input_tensor = input_image
    B, C, H, W = input_tensor.shape
    batch_size, num_parts, Hh, Ww = attention_maps.shape
    # attention_maps = torch.nn.functional.interpolate(attention_maps,size=(W,H),mode='bilinear')
    ret_imgs = []
    # print(part_weights[3])
    for i in range(batch_size):
        attention_map = attention_maps[i]
        # print(attention_map.shape)
        mask = attention_map.mean(dim=0)
        # print(type(mask))
        # mask = (mask-mask.min())/(mask.max()-mask.min())
        # threshold = random.uniform(0.4, 0.6)
        threshold = 0.1
        max_activate = mask.max()
        min_activate = threshold * max_activate
        itemindex = torch.nonzero(mask >= min_activate)

        padding_h = max(int(0.05 * H), 1)
        padding_w = max(int(0.05 * W), 1)
        height_min = itemindex[:, 0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[:, 0].max() + padding_h
        width_min = itemindex[:, 1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[:, 1].max() + padding_w
        # print(height_min,height_max,width_min,width_max)
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        # print(out_img.shape)
        ret_imgs.append(out_img)
    ret_imgs = torch.stack(ret_imgs)
    # print(ret_imgs.shape)
    return ret_imgs


# use AttentionHead as well as TwoMLHead; AttentionHead for classification, TwoMLHead for box regression
class BAP(nn.Module):
    def __init__(self, **kwargs):
        super(BAP, self).__init__()

    def forward(self, feature_maps, attention_maps):
        feature_shape = feature_maps.size()  ## N*Cf*7*7*
        attention_shape = attention_maps.size()  ## N*Ca*7*7
        # print(feature_shape,attention_shape)
        phi_I = torch.einsum('imjk,injk->imn', (attention_maps, feature_maps))  ## N*Ca*Cf
        phi_I = torch.div(phi_I, float(attention_shape[2] * attention_shape[3]))
        phi_I = torch.mul(torch.sign(phi_I), torch.sqrt(torch.abs(phi_I) + 1e-12))
        phi_I = phi_I.view(feature_shape[0], -1)  # (N, Ca*Cf)
        raw_features = torch.nn.functional.normalize(phi_I, dim=-1)  ##N*(Ca*Cf)
        pooling_features = raw_features * 100
        # print(pooling_features.shape)

        return raw_features, pooling_features


class AttentionHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

        self.attention_conv = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=3, stride=1, padding=1, bias=False)
        self.attention_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bap = BAP()

        # self.fc6 = nn.Linear(in_channels * out_channels, 1024)
        # self.fc7 = nn.Linear(1024, 1024)

        self.register_buffer('ck_buffer', torch.zeros(num_classes, in_channels * out_channels))

    def ck_loss(self, x, labels):
        x = F.normalize(x.flatten(1), dim=1)
        labels = torch.cat(labels, dim=0)
        # x = x[labels!=0]
        # labels = labels[labels!=0]
        ck_buffer = self.ck_buffer[labels]
        ck_buffer = F.normalize(ck_buffer.flatten(1), dim=1)

        ck_loss = ((x - ck_buffer) ** 2).sum() / x.size(0)

        with torch.no_grad():
            for label in torch.unique(labels):
                self.ck_buffer[label] = self.ck_buffer[label] * 0.95 + x[labels == label].mean(0) * 0.05

        return ck_loss

    def forward(self, x):
        attention_map = self.relu(self.attention_bn(self.attention_conv(x)))
        # relu, so >= 0
        # (N, output_channels, H, W)
        raw_x, x = self.bap(x, attention_map)
        # (N, out_channels * in_channels)
        # x = self.relu(self.fc7(self.relu(self.fc6(x))))

        return attention_map, raw_x, x


class AttentionHeadTransformer(AttentionHead):
    def __init__(self, in_channels, out_channels, num_classes=4):
        super().__init__(in_channels, out_channels, num_classes)
        transformer_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

    def forward(self, x):
        attention_map, raw_x, x = super().forward(x)

        x = F.relu(self.transformer_encoder(x.view(-1, self.out_channels, self.in_channels))).flatten(1)

        return attention_map, raw_x, x


class RoIHeadsN(RoIHeadsOld):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 use_focal_loss=False,
                 focal_gamma=2,
                 ):
        super().__init__(box_roi_pool,
                         box_head,
                         box_predictor,
                         # Faster R-CNN training
                         fg_iou_thresh, bg_iou_thresh,
                         batch_size_per_image, positive_fraction,
                         bbox_reg_weights,
                         # Faster R-CNN inference
                         score_thresh,
                         nms_thresh,
                         detections_per_img,
                         # Mask
                         mask_roi_pool,
                         mask_head,
                         mask_predictor,
                         keypoint_roi_pool,
                         keypoint_head,
                         keypoint_predictor)

        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        if not use_focal_loss:
            self.cls_loss_func = nn.CrossEntropyLoss()
        else:
            self.cls_loss_func = FocalLoss(gamma=focal_gamma)

    def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets,
                cls_loss_func=self.cls_loss_func)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses


class RoIHeads(RoIHeadsOld):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_head_attention,
                 box_predictor,
                 box_predictor_attention,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 use_focal_loss=False,
                 focal_gamma=2,
                 use_attention=False,
                 ):
        super().__init__(box_roi_pool,
                         box_head,
                         box_predictor,
                         # Faster R-CNN training
                         fg_iou_thresh, bg_iou_thresh,
                         batch_size_per_image, positive_fraction,
                         bbox_reg_weights,
                         # Faster R-CNN inference
                         score_thresh,
                         nms_thresh,
                         detections_per_img,
                         # Mask
                         mask_roi_pool,
                         mask_head,
                         mask_predictor,
                         keypoint_roi_pool,
                         keypoint_head,
                         keypoint_predictor)

        self.use_attention = use_attention
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        if not use_focal_loss:
            self.cls_loss_func = FocalLoss(gamma=0)
        else:
            self.cls_loss_func = FocalLoss(gamma=focal_gamma)
        self.box_head_attention = box_head_attention
        self.box_predictor_attention = box_predictor_attention
        self.use_aug = False

    def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features_0 = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features_0)
        if not self.use_attention:
            class_logits, box_regression = self.box_predictor(box_features)
        else:
            attention_map, raw_box_features_attention, box_features_attention = self.box_head_attention(box_features_0)
            class_logits, box_regression = self.box_predictor_attention(box_features, box_features_attention)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets, self.cls_loss_func)
            # for aug images
            if self.use_attention and self.use_aug:
                box_features_0_crop, box_features_0_drop = attention_crop_drop(attention_map, box_features_0)
                _, _, box_features_attention1 = self.box_head_attention(box_features_0_crop)
                class_logits1, box_regression1 = self.box_predictor_attention(box_features, box_features_attention1)
                _, _, box_features_attention2 = self.box_head_attention(box_features_0_drop)
                class_logits2, box_regression2 = self.box_predictor_attention(box_features, box_features_attention2)

                loss_classifier1, _ = fastrcnn_loss(
                    class_logits1, box_regression1, labels, regression_targets,
                    self.cls_loss_func)
                loss_classifier2, _ = fastrcnn_loss(
                    class_logits2, box_regression2, labels, regression_targets,
                    self.cls_loss_func)
                loss_classifier = (loss_classifier + loss_classifier1 + loss_classifier2) / 3

            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }
            if self.use_attention:
                loss_ck = self.box_head_attention.ck_loss(raw_box_features_attention, labels)
                losses['loss_ck'] = loss_ck
        else:
            if self.use_attention and self.use_aug:
                refined_input = mask2bbox(attention_map, box_features_0)
                _, _, box_features_attention1 = self.box_head_attention(refined_input)
                class_logits1, _ = self.box_predictor_attention(box_features, box_features_attention1)
                boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals,
                                                                    image_shapes, class_logits1)
            else:
                boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals,
                                                                    image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes, class_logits1=None):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if class_logits1 is None:
            pred_scores = F.softmax(class_logits, -1)
        else:
            pred_scores = (F.softmax(class_logits, dim=-1) + F.softmax(class_logits1, dim=-1)) / 2

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


class FastRCNNPredictorAttention(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN with AttentionHead.
    Arguments:
        in_channels_two_ml_head (int): number of input channels from TwoMLPHead
        in_channels_attention_head (int): number of input channels from AttentionHead
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels_two_ml_head, in_channels_attention_head, num_classes):
        super(FastRCNNPredictorAttention, self).__init__()
        self.cls_score = nn.Linear(in_channels_attention_head, num_classes)
        self.bbox_pred = nn.Linear(in_channels_two_ml_head, num_classes * 4)

    def forward(self, x_two_ml_head, x_attention_head):
        for x in [x_two_ml_head, x_attention_head]:
            if x.dim() == 4:
                assert list(x.shape[2:]) == [1, 1]

        x_attention_head = x_attention_head.flatten(start_dim=1)
        scores = self.cls_score(x_attention_head)

        x_two_ml_head = x_two_ml_head.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x_two_ml_head)

        return scores, bbox_deltas