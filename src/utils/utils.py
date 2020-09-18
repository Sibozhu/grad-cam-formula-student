from __future__ import division

import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def average_precision(tp, conf, n_gt):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        n_gt:  Number of ground truth objects. Always positive
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    _, i = torch.sort(-conf)
    tp, conf = tp[i].type(torch.float), conf[i].type(torch.float)

    # Accumulate FPs and TPs
    fpc = torch.cumsum(1 - tp, dim=0)
    tpc = torch.cumsum(tp, dim=0)

    # Recall
    recall_curve = tpc / (n_gt + 1e-16)
    r = (tpc[-1] / (n_gt + 1e-16))

    # Precision
    precision_curve = tpc / (tpc + fpc)
    p = tpc[-1] / (tpc[-1] + fpc[-1])

    # AP from recall-precision curve
    ap = compute_ap(recall_curve, precision_curve)

    return ap, r, p

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = torch.cat((torch.zeros((1, ), device=recall.device, dtype=recall.dtype),
                      recall,
                      torch.ones((1, ), device=recall.device, dtype=recall.dtype)))
    mpre = torch.cat((torch.zeros((1, ), device=precision.device, dtype=precision.dtype),
                      precision,
                      torch.zeros((1, ), device=precision.device, dtype=precision.dtype)))

    # compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = torch.nonzero(mrec[1:] != mrec[:-1])

    # and sum (\Delta recall) * prec
    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12g %12g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes.
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-12)

    return iou


def build_targets(target, anchors, num_anchors, num_classes, grid_size, ignore_thres):
    n_b = target.size(0)
    n_a = num_anchors
    n_c = num_classes
    n_g = grid_size
    mask = torch.zeros(n_b, n_a, n_g, n_g, dtype=torch.uint8, device=target.device)
    conf_mask = torch.ones(n_b, n_a, n_g, n_g, dtype=torch.uint8, device=target.device)
    tx = torch.zeros(n_b, n_a, n_g, n_g, dtype=torch.float, device=target.device)
    ty = torch.zeros(n_b, n_a, n_g, n_g, dtype=torch.float, device=target.device)
    tw = torch.zeros(n_b, n_a, n_g, n_g, dtype=torch.float, device=target.device)
    th = torch.zeros(n_b, n_a, n_g, n_g, dtype=torch.float, device=target.device)
    tconf = torch.zeros(n_b, n_a, n_g, n_g, dtype=torch.float, device=target.device)
    tcls = torch.zeros(n_b, n_a, n_g, n_g, n_c, dtype=torch.uint8, device=target.device)

    master_mask = torch.sum(target, dim=2) > 0

    # Convert to position relative to box
    gx = target[:, :, 1] * n_g
    gy = target[:, :, 2] * n_g
    gw = target[:, :, 3] * n_g
    gh = target[:, :, 4] * n_g

    # Get grid box indices
    gi = gx.long()
    gj = gy.long()
    # setting the excess to the first row will ensure excess rows represent a valid row,
    # since all images have at least one target
    gi[~master_mask] = gi[:, 0].unsqueeze(1).expand(*gi.shape)[~master_mask]
    gj[~master_mask] = gj[:, 0].unsqueeze(1).expand(*gj.shape)[~master_mask]
    gx[~master_mask] = gx[:, 0].unsqueeze(1).expand(*gx.shape)[~master_mask]
    gy[~master_mask] = gy[:, 0].unsqueeze(1).expand(*gy.shape)[~master_mask]
    gw[~master_mask] = gw[:, 0].unsqueeze(1).expand(*gw.shape)[~master_mask]
    gh[~master_mask] = gh[:, 0].unsqueeze(1).expand(*gh.shape)[~master_mask]

    # Get shape of gt box
    a = torch.zeros((target.shape[0], target.shape[1], 2), dtype=torch.float, device=target.device)
    b = torch.unsqueeze(gw, -1)
    c = torch.unsqueeze(gh, -1)
    gt_box = torch.cat((a, b, c), dim=2)
    # Get shape of anchor box
    anchor_shapes = torch.cat((torch.zeros((anchors.shape[0], 2), device=target.device, dtype=torch.float), anchors), 1)
    # Calculate iou between gt and anchor shapes
    gt_box_1 = torch.unsqueeze(gt_box, 2).expand(-1, -1, anchor_shapes.shape[0], -1)
    anchor_shapes_1 = anchor_shapes.view(1, 1, anchor_shapes.shape[0], anchor_shapes.shape[1]).expand(*gt_box_1.shape)
    anch_ious = bbox_iou(gt_box_1, anchor_shapes_1).permute(0,2,1)  # put in same order as conf_mask

    # Where the overlap is larger than threshold set mask to zero (ignore)
    # when the condition is false, change the index to the (ignored) last row
    gj_mask = gj.unsqueeze(1).expand(-1, num_anchors, -1)[anch_ious > ignore_thres]
    gi_mask = gi.unsqueeze(1).expand(-1, num_anchors, -1)[anch_ious > ignore_thres]
    conf_mask[:, :, gj_mask, gi_mask] = 0
    # Find the best matching anchor box
    best_n = torch.argmax(anch_ious, dim=1)

    img_dim = torch.arange(0, n_b, device=target.device).view(-1, 1).expand(*best_n.shape)

    # Masks
    mask[img_dim, best_n, gj, gi] = 1
    conf_mask[img_dim, best_n, gj, gi] = 1
    # Coordinates
    tx[img_dim, best_n, gj, gi] = gx - gi.float()
    ty[img_dim, best_n, gj, gi] = gy - gj.float()
    # Width and height
    tw[img_dim, best_n, gj, gi] = torch.log(gw / anchors[best_n, 0] + 1e-16)
    th[img_dim, best_n, gj, gi] = torch.log(gh / anchors[best_n, 1] + 1e-16)
    # One-hot encoding of label
    target_label = target[:, :, 0].long()
    tcls[img_dim, best_n, gj, gi, target_label] = 1
    tconf[img_dim, best_n, gj, gi] = 1

    return mask, conf_mask, tx, ty, tw, th, tconf, tcls
