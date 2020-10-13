# original source here: https://github.com/hollance/BlazeFace-PyTorch


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class BlazeFace(nn.Module):
    

    def __init__(self):
        super(BlazeFace, self).__init__()

        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            BlazeBlock(28, 32, stride=2),
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            BlazeBlock(42, 48, stride=2),
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(88, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )

        self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

        self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
        self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, x):

        x = F.pad(x, (1, 2, 1, 2), "constant", 0)

        b = x.shape[0]  

        x = self.backbone1(x)  # (b, 88, 16, 16)
        h = self.backbone2(x)  # (b, 96, 8, 8)



        c1 = self.classifier_8(x)  # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)  # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)  # (b, 512, 1)

        c2 = self.classifier_16(h)  # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)  # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)  # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)  # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)  # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)  # (b, 512, 16)

        r2 = self.regressor_16(h)  # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)  # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)  # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return [r, c]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path):
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        assert (self.anchors.ndimension() == 2)
        assert (self.anchors.shape[0] == self.num_anchors)
        assert (self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 128
        assert x.shape[3] == 128

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17))
            filtered_detections.append(faces)

        return filtered_detections

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):

        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clip(-thresh, thresh)
        detection_scores = sigmoid(raw_score_tensor).squeeze(-1)

        mask = detection_scores >= self.min_score_thresh

        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = np.expand_dims(detection_scores[i, mask[i]], -1)
            output_detections.append(np.concatenate((boxes, scores), -1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        boxes = np.zeros_like(raw_boxes)
        anchors = anchors.numpy()

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(6):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 17).

        Returns a list of PyTorch tensors, one for each detected face.
        
        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = np.argsort(detections[:, 16])[::-1]

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.copy()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(axis=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections

    # IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py


def intersect(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(np.broadcast_to(np.expand_dims(box_a[:, 2:], 1), (A, B, 2)),
                    np.broadcast_to(np.expand_dims(box_b[:, 2:], 0), (A, B, 2)))
    min_xy = np.maximum(np.broadcast_to(np.expand_dims(box_a[:, :2], 1), (A, B, 2)),
                    np.broadcast_to(np.expand_dims(box_b[:, :2], 0), (A, B, 2)))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=1)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = np.broadcast_to(np.expand_dims((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]), 1), inter.shape)  # [A,B]
    area_b = np.broadcast_to(np.expand_dims((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1]), 0), inter.shape)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(np.expand_dims(box, 0), other_boxes).squeeze(0)
