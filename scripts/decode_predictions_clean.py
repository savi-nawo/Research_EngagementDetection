import torch
import torchvision

def decode_predictions_clean(outputs, anchors,
                             conf_threshold=0.10,   # LOWER threshold = more stable
                             nms_iou=0.5):

    device = outputs[0].device
    boxes = []
    scores = []

    scales = ["small", "medium", "large"]

    for s in range(3):
        out = outputs[s][0]  # (A, C, H, W)
        A, C, H, W = out.shape

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        anchors_tensor = torch.tensor(anchors[scales[s]], device=device).float()

        for a in range(A):
            tx = out[a, 0]
            ty = out[a, 1]
            tw = out[a, 2]
            th = out[a, 3]
            obj = torch.sigmoid(out[a, 4])

            mask = obj > conf_threshold
            if mask.sum() == 0:
                continue

            aw, ah = anchors_tensor[a]

            bx = (torch.sigmoid(tx) + grid_x) / W
            by = (torch.sigmoid(ty) + grid_y) / H

            bw = torch.exp(tw) * aw
            bh = torch.exp(th) * ah

            x1 = bx - bw/2
            y1 = by - bh/2
            x2 = bx + bw/2
            y2 = by + bh/2

            boxes.append(torch.stack([x1[mask], y1[mask], x2[mask], y2[mask]], 1))
            scores.append(obj[mask])

    if len(boxes) == 0:
        return [], []

    boxes = torch.cat(boxes)
    scores = torch.cat(scores)

    keep = torchvision.ops.nms(boxes, scores, nms_iou)

    return boxes[keep].cpu(), scores[keep].cpu()
