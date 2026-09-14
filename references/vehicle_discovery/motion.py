# 修改时间：2026-09-13。
# 修改目的：修正连续三次检测被错误计为四次残差候选的需求偏差。
# 修改内容：首次残差候选计数由零改为一，其余算法和冻结参数保持原样。
# 修改时间：2026-09-13。
# 修改目的：针对 tune 中旧位置拖影抢关联和纹理误报完成最后一轮调整。
# 修改内容：保留原始残差噪声阈值并加入速度预测关联与三像素最小位移。
# 修改时间：2026-09-13。
# 修改目的：针对 tune 中背景亚像素误差和车辆边缘碎裂完成第一轮调整。
# 修改内容：增加局部一像素灰度残差容差并缩窄共同边缘删除范围。
# 修改时间：2026-09-13。
# 修改目的：验证仅凭连续图像发现动态车辆的离线可行性。
# 修改内容：实现背景单应配准、公共区域残差候选与连续三次残余位移关联。
"""纯图像检测器；标签、姿态和文件路径均不进入检测接口。"""
import cv2
import numpy as np


DEFAULT_CONFIG = {
    'interval': 0.3, 'orb_features': 4000, 'orb_fast': 12,
    'match_ratio': 0.75, 'ransac_px': 2.5, 'min_inliers': 30,
    'min_inlier_ratio': 0.45, 'max_median_error': 1.5,
    'min_spatial_coverage': 0.12, 'min_overlap': 0.30,
    'border': 12, 'blur': 3, 'residual_threshold': 18,
    'noise_mad_factor': 4.0, 'remove_shared_edges': True,
    'residual_tolerance_px': 1, 'edge_dilation': 1,
    'threshold_from_original': True, 'velocity_association': True,
    'min_area': 10, 'max_area': 1800, 'min_side': 4,
    'max_side': 90, 'max_aspect': 5.0, 'padding': 3,
    'min_residual_px': 3.0, 'max_residual_px': 30.0,
    'max_area_ratio': 3.5, 'min_direction_cos': 0.5,
    'stable_steps': 3, 'max_gap_s': 0.85, 'seed': 20260913,
}


class MotionDetector:
    def __init__(self, config):
        self.config = DEFAULT_CONFIG | dict(config)
        cv2.setNumThreads(2)
        self.orb = cv2.ORB_create(nfeatures=self.config['orb_features'],
                                  fastThreshold=self.config['orb_fast'])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    def reset(self):
        self.previous = None
        self.tracks = []
        self.last_debug = {}
        cv2.setRNGSeed(self.config['seed'])

    def _skip(self, reason, diagnostics):
        # 失败帧仍成为下一次配准参考，同时切断连续性，允许之后重新恢复。
        self.tracks = []
        return {'candidates': [], 'raw_candidates': [], 'processed': False,
                'reason': reason, 'diagnostics': diagnostics}

    @staticmethod
    def _transform(points, matrix):
        return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)

    def detect(self, rgb, time):
        cfg = self.config
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        old = self.previous
        self.previous = (gray, keypoints, descriptors, float(time))
        self.last_debug = {}
        diagnostics = {'keypoints': len(keypoints)}
        if old is None:
            return self._skip('first_frame', diagnostics)
        previous_gray, previous_kp, previous_des, previous_time = old
        dt = float(time) - previous_time
        diagnostics['dt'] = dt
        if dt <= 0 or dt > cfg['max_gap_s']:
            return self._skip('time_gap', diagnostics)
        if previous_gray.shape != gray.shape:
            return self._skip('shape_change', diagnostics)
        if descriptors is None or previous_des is None or len(descriptors) < 2:
            return self._skip('no_descriptors', diagnostics)
        pairs = self.matcher.knnMatch(previous_des, descriptors, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < cfg['match_ratio'] * pair[1].distance]
        # 去掉多个旧特征争抢同一当前特征的匹配。
        unique = {}
        for match in sorted(matches, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        diagnostics['matches'] = len(matches)
        if len(matches) < cfg['min_inliers']:
            return self._skip('few_matches', diagnostics)
        src = np.float32([previous_kp[m.queryIdx].pt for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, cfg['ransac_px'])
        if matrix is None or not np.isfinite(matrix).all() or inliers is None:
            return self._skip('homography_failed', diagnostics)
        good = inliers.ravel().astype(bool)
        count = int(good.sum())
        ratio = count / len(matches)
        error = np.linalg.norm(self._transform(src, matrix) - dst, axis=1)
        median = float(np.median(error[good])) if count else 999.0
        height, width = gray.shape
        coverage = float(cv2.contourArea(cv2.convexHull(dst[good]))) / (width * height) if count >= 3 else 0.0
        diagnostics.update(inliers=count, inlier_ratio=ratio, median_error=median, spatial_coverage=coverage)
        if count < cfg['min_inliers'] or ratio < cfg['min_inlier_ratio'] or median > cfg['max_median_error'] or coverage < cfg['min_spatial_coverage']:
            return self._skip('registration_quality', diagnostics)
        corners = self._transform([[0, 0], [width, 0], [width, height], [0, height]], matrix)
        area_ratio = abs(float(cv2.contourArea(corners))) / (width * height)
        if not np.isfinite(corners).all() or not cv2.isContourConvex(corners) or not 0.35 < area_ratio < 3.0:
            return self._skip('invalid_geometry', diagnostics)
        mask = cv2.warpPerspective(np.full_like(gray, 255), matrix, (width, height), flags=cv2.INTER_NEAREST)
        border = cfg['border']
        mask = cv2.erode(mask, np.ones((2 * border + 1, 2 * border + 1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        overlap = float(np.mean(mask > 0))
        diagnostics.update(overlap=overlap, homography=matrix.tolist())
        if overlap < cfg['min_overlap']:
            return self._skip('low_overlap', diagnostics)
        warped = cv2.warpPerspective(previous_gray, matrix, (width, height))
        blurred = cv2.GaussianBlur(gray, (cfg['blur'], cfg['blur']), 0)
        warped_blur = cv2.GaussianBlur(warped, (cfg['blur'], cfg['blur']), 0)
        signed = blurred.astype(np.float32) - warped_blur.astype(np.float32)
        offset = float(np.median(signed[mask > 0]))
        residual = np.abs(signed - offset)
        original_values = residual[mask > 0]
        radius = cfg['residual_tolerance_px']
        if radius:
            # 仅容忍局部配准误差，不以标签或姿态估计背景运动。
            padded = cv2.copyMakeBorder(warped_blur, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
            for dy in range(2 * radius + 1):
                for dx in range(2 * radius + 1):
                    delta = np.abs(blurred.astype(np.float32) - padded[dy:dy+height, dx:dx+width].astype(np.float32) - offset)
                    residual = np.minimum(residual, delta)
        values = residual[mask > 0]
        noise_values = original_values if cfg['threshold_from_original'] else values
        noise = float(np.median(noise_values))
        mad = float(np.median(np.abs(noise_values - noise)))
        threshold = max(cfg['residual_threshold'], noise + cfg['noise_mad_factor'] * 1.4826 * mad)
        shared_edges = np.zeros_like(gray)
        if cfg['remove_shared_edges']:
            edges = cv2.Canny(gray, 60, 140)
            old_edges = cv2.Canny(warped, 60, 140)
            shared_edges = cv2.bitwise_and(edges, cv2.dilate(old_edges, np.ones((3, 3), np.uint8)))
            shared_edges = cv2.dilate(shared_edges, np.ones((cfg['edge_dilation'], cfg['edge_dilation']), np.uint8))
        binary = ((residual >= threshold) & (mask > 0) & (shared_edges == 0)).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        binary[mask == 0] = 0
        n, components, stats, centers = cv2.connectedComponentsWithStats(binary)
        candidates = []
        for index in range(1, n):
            x, y, w, h, area = map(int, stats[index])
            if not cfg['min_area'] <= area <= cfg['max_area']:
                continue
            if min(w, h) < cfg['min_side'] or max(w, h) > cfg['max_side'] or max(w / h, h / w) > cfg['max_aspect']:
                continue
            pad = cfg['padding']
            strength = float(residual[y:y+h, x:x+w][components[y:y+h, x:x+w] == index].mean())
            candidates.append({'box': [max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad)],
                               'score': min(1.0, strength / 80.0)})
        tracks, stable = self._associate(candidates, matrix, dt)
        self.tracks = tracks
        diagnostics.update(threshold=float(threshold), residual_median=noise,
                           residual_p95=float(np.percentile(values, 95)), residual_fraction=float(np.mean(binary[mask > 0] > 0)),
                           raw_count=len(candidates), stable_count=len(stable),
                           track_steps=[t['steps'] for t in tracks])
        self.last_debug = {'gray': gray, 'warped': warped, 'residual': residual, 'mask': mask,
                           'binary': binary, 'shared_edges': shared_edges}
        return {'candidates': stable, 'raw_candidates': candidates, 'processed': True,
                'reason': 'ok', 'diagnostics': diagnostics}

    def _associate(self, candidates, matrix, dt):
        cfg = self.config
        tracks = [{'candidate': c, 'center': [(c['box'][0]+c['box'][2])/2, (c['box'][1]+c['box'][3])/2],
                   'steps': 1, 'velocity': None} for c in candidates]
        edges = []
        for previous_index, old in enumerate(self.tracks):
            box = old['candidate']['box']
            transformed = self._transform([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]], old['center']], matrix)
            predicted = transformed[-1]
            old_area = max(1., abs(float(cv2.contourArea(transformed[:4]))))
            mapped_velocity = None
            if old['velocity'] is not None:
                endpoint = np.asarray(old['center']) + np.asarray(old['velocity'])
                mapped_velocity = self._transform([endpoint], matrix)[0] - predicted
            for index, track in enumerate(tracks):
                delta = np.asarray(track['center']) - predicted
                distance = float(np.linalg.norm(delta))
                if not cfg['min_residual_px'] * dt / .3 <= distance <= cfg['max_residual_px'] * dt / .3:
                    continue
                b = track['candidate']['box']
                area = (b[2]-b[0]) * (b[3]-b[1])
                if max(area/old_area, old_area/max(1, area)) > cfg['max_area_ratio']:
                    continue
                if mapped_velocity is not None:
                    cosine = float(np.dot(delta, mapped_velocity) / max(1e-6, np.linalg.norm(mapped_velocity) * distance))
                    if cosine < cfg['min_direction_cos']:
                        continue
                cost = distance
                if cfg['velocity_association'] and mapped_velocity is not None:
                    # 已有速度的候选优先沿预测位置续接，避免旧位置拖影抢占。
                    cost = float(np.linalg.norm(delta - mapped_velocity * dt))
                    if cost > max(4.0, 0.7 * float(np.linalg.norm(mapped_velocity)) * dt):
                        continue
                edges.append((cost, previous_index, index, delta / dt))
        used_old, used_new = set(), set()
        for _, old_index, index, velocity in sorted(edges, key=lambda e: e[0]):
            if old_index in used_old or index in used_new:
                continue
            used_old.add(old_index)
            used_new.add(index)
            tracks[index]['steps'] = self.tracks[old_index]['steps'] + 1
            tracks[index]['velocity'] = velocity.tolist()
        stable = [t['candidate'].copy() for t in tracks if t['steps'] >= cfg['stable_steps']]
        return tracks, stable


def build_detector(config):
    return MotionDetector(config)
