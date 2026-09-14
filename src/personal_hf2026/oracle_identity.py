# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13（H1 原生锁定去重）
# 修改目的：避免同一车辆的原生坐标和理想坐标被当成两个锁定竞争者。
# 修改内容：按已唯一确认的车辆 ID 合并主锁定对应候选，并保存去重审计。
# 修改时间：2026-09-13
# 修改目的：仅为数据采集提供真值类别接纳，保留原生检测与完整竞争车辆坐标。
# 修改内容：新增唯一位置身份关联、无速度拒绝的确认策略以及专用智能体。
"""采集专用真值类别策略；不用于正式比赛或视觉算法成绩。"""
from dataclasses import asdict, replace
import math

from .control_test_runner import _ground_distance_m, visible_vehicle_detections
from .personal_v1 import PersonalV1Agent
from .search_filter import MultiCandidateSearch, SearchFilterResult
from .tracking import LocalTrackManager, estimate_track_velocity


def identify_position(position, vehicles, tolerance_m=1.0):
    """只接受容差内唯一车辆；多车重叠时保留未知，不选最近类别。"""
    matches = list(dict.fromkeys((uid, kind) for uid, kind, lat, lon in vehicles
               if _ground_distance_m(*position, lat, lon) <= tolerance_m))
    return matches[0] if len(matches) == 1 else (None, "unknown")


def oracle_detections(ws, uid):
    """原生检测只补类别，同 ID 候选合并；其它车辆与未知身份的歧义均保留。"""
    from competition.sdk.core.isolation import _extract_truth
    primary = _extract_truth(ws.entities[uid])
    vehicles = [(key, kind, e.lat, e.lon)
                for collection, kind in ((ws.targets, "TargetVehicle"), (ws.decoys, "DecoyVehicle"))
                for key, e in collection.items()]
    result, audit = [], []
    for index, detection in enumerate([primary, *visible_vehicle_detections(ws, uid)]):
        identity, kind = None, "unknown"
        if detection.detected and detection.target_lat is not None and detection.target_lon is not None:
            identity, kind = identify_position((detection.target_lat, detection.target_lon), vehicles,
                                               1.0 if index == 0 else 0.001)
        if index > 0 and identity is not None and identity == audit[0]["vehicle_id"]:
            # 仅合并已经唯一确认的同一车辆；主检测的 detected、坐标和锁定事实原样保留。
            audit[0]["deduplicated_native_uid"] = identity
            audit[0].setdefault("deduplicated_ideal_candidates", []).append(asdict(detection))
            continue
        result.append(replace(detection, target_type=kind))
        audit.append(dict(source="native_primary" if index == 0 else "ideal_candidate",
                          vehicle_id=identity, identity=kind,
                          original=asdict(detection), forwarded=asdict(result[-1])))
    return result, audit


class OracleCandidateSearch(MultiCandidateSearch):
    """输入已按真值类别筛选，仅等待普通坐标轨迹确认，不使用速度判断真假。"""
    def update(self, now, positions):
        self._associate(now, positions)
        self._update_view_center(now)
        snapshots = [(c.track_id, c.manager.snapshot(now, window_s=6.0)) for c in self._tracks]
        fresh = [(key, s) for key, s in snapshots if s.points and abs(s.last_seen - now) < 1e-6]
        confirmed = [(key, s) for key, s in fresh if s.state == LocalTrackManager.CONFIRMED]
        selected = confirmed or fresh
        if not selected:
            self.status = self.PENDING
            self.focus_position = None
            self.last_result = SearchFilterResult(self.PENDING, 0.0, "no_oracle_target")
            return None
        key, snapshot = max(selected, key=lambda pair: now - pair[1].points[0].t)
        self.epoch, self.started_at = snapshot.epoch, snapshot.points[0].t
        self.focus_position = snapshot.position
        speed = math.hypot(*estimate_track_velocity(snapshot.points))
        self.status = self.ACCEPTED if confirmed else self.PENDING
        self.last_result = SearchFilterResult(self.status, speed,
                                             "oracle_true" if confirmed else "coordinate_confirmation")
        if confirmed:
            self.accepted_count += 1
            self.accepted_track_id = key
            return snapshot
        return None


class OracleIdentityAgent(PersonalV1Agent):
    """身份限制同时覆盖搜索、已锁跟随、FOLLOWER 初次绑定和后续重关联。"""
    def reset(self):
        super().reset()
        self._search_filter = OracleCandidateSearch()

    def _eligible_positions(self, detections, positions):
        eligible = [(d.target_lat, d.target_lon) for d in detections
                    if d.detected and d.target_type == "TargetVehicle"
                    and d.target_lat is not None and d.target_lon is not None]
        return tuple(dict.fromkeys(eligible))

    def _competition_positions(self, eligible_positions, all_positions):
        return all_positions
