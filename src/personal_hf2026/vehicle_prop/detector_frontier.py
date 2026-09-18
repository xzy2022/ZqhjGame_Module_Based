# 修改时间：2026-09-18
# 修改目的：把队友交付的 FrontierPipeline 迁入可安装的 personal_hf2026 包。
# 修改内容：改为包内相对导入和项目固定配置、权重路径，并保留检测与跟踪算法逻辑。
"""Two-class YOLO26 detection with score-preserving view fusion and causal tracking."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from personal_hf2026.paths import PROJECT_ROOT

from .frontier_runtime import CONFIG_PATH, activate
activate()
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import nms
from .detector_improved import suppress, tile_windows
from .temporal_tracker import CameraMotion, TemporalTracker, overlaps


class DualScorePredictor(DetectionPredictor):
    """Retain both class scores from exactly the anchors selected by NMS."""
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        if raw.ndim != 3 or raw.shape[1] != 6 or getattr(self.model, 'end2end', False):
            raise ValueError('Expected a two-class one-to-many detection head; use nms=True')
        scores = raw[:, 4:6].transpose(1, 2).clone()
        boxes, indices = nms.non_max_suppression(raw, self.args.conf, self.args.iou,
                                               agnostic=True, nc=2, max_det=self.args.max_det,
                                               return_idxs=True, max_time_img=1.)
        results = self.construct_results(boxes, img, orig_imgs)
        for result, score, kept in zip(results, scores, indices):
            result.class_scores = score[kept.reshape(-1).long()].detach().cpu().numpy()
        return results


def merge_scored(candidates, iou=.5, method='nms'):
    """Nx7 = xyxy, confidence, real/prop conditional scores. No score inflation."""
    a = np.asarray(candidates, dtype=np.float32).reshape(-1, 7)
    if not len(a):
        return a
    if method == 'nms':
        return a[suppress(a[:, :4], a[:, 4], iou)]
    if method != 'weighted':
        raise ValueError(method)
    remaining = a[np.argsort(-a[:, 4], kind='stable')]
    output = []
    while len(remaining):
        mask = overlaps(remaining[:1, :4], remaining[:, :4])[0] > iou
        mask[0] = True
        group = remaining[mask]
        weights = np.maximum(group[:, 4]**2, 1e-12)
        box = np.average(group[:, :4], axis=0, weights=weights)
        probability = np.average(group[:, 5:7], axis=0, weights=weights)
        probability /= max(probability.sum(), 1e-9)
        output.append([*box, float(group[:, 4].max()), *probability])
        remaining = remaining[~mask]
    return np.asarray(output, dtype=np.float32).reshape(-1, 7)


class FrontierDetector:
    def __init__(self, weights, device='0', imgsz=2048, flip=True, tile_size=0, tile_imgsz=1024,
                 iou=.5, merge='nms', expected_sha256=None, full_frame=True):
        self.weights = Path(weights)
        if expected_sha256 and hashlib.sha256(self.weights.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Detector weights checksum mismatch')
        self.model = YOLO(str(self.weights))
        if self.model.names != {0:'real_vehicle', 1:'model_prop'}:
            raise ValueError('Expected real_vehicle/model_prop class mapping')
        self.device, self.imgsz, self.flip = device, imgsz, flip
        self.tile_size, self.tile_imgsz = tile_size, tile_imgsz
        self.iou, self.merge = iou, merge
        self.full_frame = full_frame

    def _predict(self, images, imgsz, raw_conf):
        results = self.model.predict(images, predictor=DualScorePredictor, device=self.device,
                                     imgsz=imgsz, conf=raw_conf, iou=self.iou, agnostic_nms=True,
                                     nms=True, max_det=200, verbose=False)
        output = []
        for result in results:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            score = result.boxes.conf.detach().cpu().numpy()
            probabilities = result.class_scores
            probabilities = probabilities / np.maximum(probabilities.sum(1, keepdims=True), 1e-9)
            output.append(np.column_stack((boxes, score, probabilities)).astype(np.float32))
        return output

    def predict_components(self, images, raw_conf=.001):
        original = self._predict(images, self.imgsz, raw_conf) if self.full_frame else [np.empty((0,7),dtype=np.float32) for _ in images]
        flipped = [np.empty((0,7),dtype=np.float32) for _ in images]
        if self.flip and self.full_frame:
            flipped = self._predict([np.ascontiguousarray(im[:, ::-1]) for im in images], self.imgsz, raw_conf)
            for im, a in zip(images, flipped):
                left = im.shape[1]-a[:, 2].copy()
                a[:, 2] = im.shape[1]-a[:, 0]
                a[:, 0] = left
        tiled = []
        for im in images:
            all_tiles = []
            if self.tile_size:
                windows = tile_windows(im.shape[1], im.shape[0], self.tile_size, .25)
                patches = [im[y:b,x:r] for x,y,r,b in windows]
                for start in range(0,len(patches),6):
                    arrays = self._predict(patches[start:start+6], self.tile_imgsz, raw_conf)
                    for win, a in zip(windows[start:start+6], arrays):
                        a[:, :4] += np.array([win[0],win[1],win[0],win[1]])
                        all_tiles.extend(a.tolist())
            tiled.append(all_tiles)
        return [dict(full_original=a.tolist(), full_flipped=b.tolist(), tiles=c)
                for a,b,c in zip(original,flipped,tiled)]

    def predict(self, image, raw_conf=.001):
        components = self.predict_components([image], raw_conf)[0]
        return merge_scored(components['full_original']+components['full_flipped']+components['tiles'], self.iou, self.merge)


def arrays_for_tracker(candidates):
    a = np.asarray(candidates,dtype=np.float32).reshape(-1,7)
    return a[:, :5], np.column_stack((a[:, 5:7], np.zeros(len(a),dtype=np.float32)))


class FrontierPipeline:
    def __init__(self, config=None, device='0'):
        self.config = json.loads(Path(config or CONFIG_PATH).read_text(encoding='utf-8'))
        self.detector = FrontierDetector(PROJECT_ROOT/self.config['weights'], device=device,
                                         expected_sha256=self.config['sha256'], **self.config['detector'])
        self.tracker = TemporalTracker(**self.config['tracker'])
        self.foundation = None
        if self.config.get('foundation_classifier'):
            from foundation_classifier import FoundationClassifier
            spec=self.config['foundation_classifier']
            self.foundation=FoundationClassifier(ROOT/spec['weights'],device='cpu' if device=='cpu' else 'cuda:'+str(device),expected_sha256=spec['sha256'])
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.camera = CameraMotion()
        self.previous_t = None
        self.sequence = None

    def predict(self, image, timestamp=0., sequence_id='single'):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected BGR uint8 image')
        if self.sequence != sequence_id or (self.previous_t is not None and (timestamp<=self.previous_t or timestamp-self.previous_t>.25)):
            self.camera = CameraMotion()
        self.sequence, self.previous_t = sequence_id, timestamp
        motion, quality = self.camera.update(image)
        boxes, probabilities = arrays_for_tracker(self.detector.predict(image))
        if self.foundation is not None:
            keep=boxes[:,4]>=self.tracker.low
            boxes,probabilities=boxes[keep],probabilities[keep]
            extra=self.foundation.predict(image,boxes[:,:4])[:,:2]
            extra/=np.maximum(extra.sum(1,keepdims=True),1e-9)
            weight=self.config['foundation_classifier']['blend']
            probabilities[:,:2]=(1-weight)*probabilities[:,:2]+weight*extra
        detections = self.tracker.update(boxes, probabilities, timestamp, sequence_id, motion, image.shape[:2])
        for d in detections:
            d['class_name'] = ('real_vehicle','model_prop')[d['class_id']] if d['class_confidence'] >= self.config['unknown_threshold'] else 'uncertain'
        return detections
