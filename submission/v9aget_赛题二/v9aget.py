# 修改时间：2026-09-27。
# 修改目的：基于 V4 生成符合赛题二提交规范的独立单文件 Agent。
# 修改内容：整合控制与推理逻辑，只消费本机照片、姿态、通信和允许的仿真时间。
"""v9aget：赛题二正式提交入口，三架无人机各实例独立控制。"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from threading import Condition, Lock, Thread
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from competition.sdk.core.commands import broadcast, fly_to, point_gimbal, report_target, set_gimbal_fov
from competition.sdk.core.observation import Detection
from competition.sdk.scenarios.coop_decoy import CoopAgent


# 任务区域来自官方赛题二基线中的静态地图边界，不读取场景或航线文件。
_BBOX = ((26.982, 124.980), (27.025, 125.020))


def _haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0



# ===== V4 来源：visual_geometry.py =====
@dataclass(frozen=True)
class PixelBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    width: int
    height: int
    category: str = "vehicle_candidate"
    class_margin: float = 0.0

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

def angle_delta(a, b):
    return (a - b + 180) % 360 - 180

def aligned_sample(history, t, max_gap=0.35):
    """只在历史覆盖的时间内插值；候选必须在两端均存在。"""
    for row in history:
        if abs(row["t"] - t) < 1e-5:
            return row, "exact"
    for a, b in zip(history, history[1:]):
        if not a["t"] <= t <= b["t"]:
            continue
        dt = b["t"] - a["t"]
        if dt > max_gap or dt <= 0:
            return None, "pose_gap"
        if abs(a["own"]["gimbal_fov_deg"] - b["own"]["gimbal_fov_deg"]) > 0.2:
            return None, "fov_transition"
        ratio = (t - a["t"]) / dt
        own = {}
        for key, value in a["own"].items():
            delta = b["own"][key] - value
            if key in ("heading_deg", "gimbal_pan"):
                delta = angle_delta(b["own"][key], value)
            own[key] = value + ratio * delta
        common = a["tracks"].keys() & b["tracks"].keys()
        tracks = {key: [a["tracks"][key][j] + ratio *
                       (b["tracks"][key][j] - a["tracks"][key][j]) for j in (0, 1)]
                  for key in common}
        rate = max(abs(angle_delta(b["own"][k], a["own"][k])) / dt
                   for k in ("heading_deg", "gimbal_pan", "gimbal_tilt"))
        return {"t": t, "own": own, "tracks": tracks, "angular_rate_dps": rate}, "interpolated"
    return None, "time_not_bracketed"

def pixel_ray(u, v, width, height, own):
    focal = width / (2 * math.tan(math.radians(own["gimbal_fov_deg"] / 2)))
    x, y = (u - (width - 1) / 2) / focal, (v - (height - 1) / 2) / focal
    yaw = math.radians(own["heading_deg"] + own["gimbal_pan"])
    pitch = math.radians(own["gimbal_tilt"])
    forward = (math.cos(pitch) * math.sin(yaw), math.cos(pitch) * math.cos(yaw), math.sin(pitch))
    right = (math.cos(yaw), -math.sin(yaw), 0)
    down = (math.sin(pitch) * math.sin(yaw), math.sin(pitch) * math.cos(yaw), -math.cos(pitch))
    return tuple(forward[i] + x * right[i] + y * down[i] for i in range(3)), focal

def bind_boxes(boxes, sample):
    """类别不参与几何选择；一框多轨或多框同轨都拒绝。"""
    own, tracks = sample["own"], sample["tracks"]
    results = []
    for index, box in enumerate(boxes):
        ray, focal = pixel_ray(*box.center, box.width, box.height, own)
        horizontal = math.hypot(ray[0], ray[1])
        row = {"box_index": index, "track_id": None, "status": "unmatched", "candidates": []}
        results.append(row)
        if sample.get("angular_rate_dps", 0) > 20:
            row["status"] = "fast_pose"
            continue
        if horizontal * focal < 12 or ray[2] >= 0:
            row["status"] = "near_nadir_or_upward"
            continue
        bearing = math.degrees(math.atan2(ray[0], ray[1]))
        radius = math.hypot(box.x2 - box.x1, box.y2 - box.y1) / 2
        tolerance = min(15.0, 2.0 + math.degrees(math.atan2(radius, horizontal * focal)))
        row.update(bearing_deg=bearing, tolerance_deg=tolerance)
        for track_id, (lat, lon) in tracks.items():
            north = (lat - own["lat"]) * 111320
            east = (lon - own["lon"]) * 111320 * math.cos(math.radians(own["lat"]))
            if math.hypot(north, east) < 15:
                continue
            error = abs(angle_delta(bearing, math.degrees(math.atan2(east, north))))
            if error <= tolerance:
                row["candidates"].append({"track_id": track_id, "error_deg": error})
        if len(row["candidates"]) == 1:
            row["track_id"] = row["candidates"][0]["track_id"]
            row["status"] = "bound"
        elif len(row["candidates"]) > 1:
            row["status"] = "ambiguous"
    counts = {}
    for row in results:
        if row["status"] == "bound":
            counts[row["track_id"]] = counts.get(row["track_id"], 0) + 1
    for row in results:
        if row["status"] == "bound" and counts[row["track_id"]] > 1:
            row.update(status="multiple_boxes", track_id=None)
    return results




# ===== V4 来源：search_route.py =====
class StripSearchRoute:
    def __init__(self, bounds, partition_index=0, partition_count=1,
                 lane_spacing_m=200.0, arrival_radius_m=60.0, grid_size_m=100.0):
        (south, west), (north, east) = bounds
        width = (east - west) / partition_count
        self.partition = ((south, west + width * partition_index),
                          (north, west + width * (partition_index + 1)))
        self.bounds = bounds
        self.lane_spacing_m = lane_spacing_m
        self.arrival_radius_m = arrival_radius_m
        self.grid_size_m = grid_size_m
        self._lon_scale = 111320.0 * math.cos(math.radians((south + north) / 2))
        self.waypoints = []
        self.index = 0
        self.direction = 1
        self.completed_waypoints = 0
        self.completed_passes = 0
        self._active = False
        self.visited_cells = set()
        self.current_cell = None
        self.cell_entries = 0
        self.repeat_entries = 0

    def _distance(self, first, second):
        return math.hypot((first[0] - second[0]) * 111320.0,
                          (first[1] - second[1]) * self._lon_scale)

    def _build(self, position):
        (south, west), (north, east) = self.partition
        intervals = max(1, math.ceil((north - south) * 111320.0 / self.lane_spacing_m))
        latitudes = [south + (north - south) * i / intervals for i in range(intervals + 1)]
        choices = []
        for rows in (latitudes, list(reversed(latitudes))):
            for start_west in (True, False):
                points = []
                for i, latitude in enumerate(rows):
                    forward = start_west == (i % 2 == 0)
                    points.extend([(latitude, west if forward else east),
                                   (latitude, east if forward else west)])
                choices.append(points)
        self.waypoints = min(choices, key=lambda points: self._distance(position, points[0]))

    def observe_search_position(self, position):
        """只按实际搜索位置记网格，不把指令航点或假设的相机足迹当作覆盖。"""
        if position is None:
            self.current_cell = None
            return
        (south, west), _ = self.bounds
        cell = (math.floor((position[1] - west) * self._lon_scale / self.grid_size_m),
                math.floor((position[0] - south) * 111320.0 / self.grid_size_m))
        if cell != self.current_cell:
            self.cell_entries += 1
            self.repeat_entries += int(cell in self.visited_cells)
            self.visited_cells.add(cell)
            self.current_cell = cell

    def pause(self):
        """候选追踪和协同期间暂停航点推进。"""
        self._active = False

    def target(self, position):
        if not self.waypoints:
            self._build(position)
        self._active = True
        if self._distance(position, self.waypoints[self.index]) <= self.arrival_radius_m:
            self.completed_waypoints += 1
            next_index = self.index + self.direction
            if not 0 <= next_index < len(self.waypoints):
                # 完成一遍后反向复查，避免跨越整个分区跳回起点。
                self.direction *= -1
                self.completed_passes += 1
                next_index = self.index + self.direction
            self.index = next_index
        return self.waypoints[self.index]

    @property
    def summary(self):
        return {
            "partition": self.partition,
            "lane_spacing_m": self.lane_spacing_m,
            "arrival_radius_m": self.arrival_radius_m,
            "waypoint_count": len(self.waypoints),
            "waypoint_index": self.index,
            "direction": self.direction,
            "completed_waypoints": self.completed_waypoints,
            "completed_passes": self.completed_passes,
            "waypoint": self.waypoints[self.index] if self.waypoints else None,
            "active": self._active,
            "grid_size_m": self.grid_size_m,
            "current_cell": self.current_cell,
            "visited_cells": len(self.visited_cells),
            "cell_entries": self.cell_entries,
            "repeat_entry_ratio": self.repeat_entries / self.cell_entries if self.cell_entries else 0.0,
        }




# ===== V4 来源：survey_search.py =====
class SearchCoverageGrid:
    def __init__(self, bounds, cell_m=100.0, relative_heights_m=(250.0, 500.0)):
        self.bounds = bounds
        (south, west), (north, east) = bounds
        self.lon_scale = 111320.0 * math.cos(math.radians((south + north) / 2))
        self.width = (east - west) * self.lon_scale
        self.height = (north - south) * 111320.0
        self.nx = max(1, math.ceil(self.width / cell_m))
        self.ny = max(1, math.ceil(self.height / cell_m))
        self.dx, self.dy = self.width / self.nx, self.height / self.ny
        self.relative_heights_m = relative_heights_m
        self.last_seen = {}
        self.new_cells = []

    def xy(self, position):
        return ((position[1] - self.bounds[0][1]) * self.lon_scale,
                (position[0] - self.bounds[0][0]) * 111320.0)

    def point(self, x, y):
        return (self.bounds[0][0] + y / 111320.0,
                self.bounds[0][1] + x / self.lon_scale)

    def observe(self, now, position, heading, pan, tilt, fov):
        """只有实际光锥包含网格四角才记覆盖，不按目标指令或航点到达记账。"""
        self.new_cells = []
        x, y = self.xy(position)
        az, el = math.radians(heading + pan), math.radians(tilt)
        axis = (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), -math.sin(el))
        cos_limit = math.cos(math.radians(max(0.1, fov / 2 - 2.0)))
        # 相对高度区间两端都满足时，中间高度也在凸光锥内；这不是目标高程估计。
        for ix in range(self.nx):
            for iy in range(self.ny):
                valid = True
                for cx, cy in ((ix*self.dx, iy*self.dy), ((ix+1)*self.dx, iy*self.dy),
                               (ix*self.dx, (iy+1)*self.dy), ((ix+1)*self.dx, (iy+1)*self.dy)):
                    ex, ey = cx-x, cy-y
                    for height in self.relative_heights_m:
                        dot = ex*axis[0] + ey*axis[1] + height*axis[2]
                        if dot < cos_limit * math.sqrt(ex*ex + ey*ey + height*height):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    cell = (ix, iy)
                    if cell not in self.last_seen:
                        self.new_cells.append(cell)
                    self.last_seen[cell] = now

    @property
    def summary(self):
        total = self.nx*self.ny
        return dict(shape=(self.nx,self.ny), covered_cells=len(self.last_seen),
                    uncovered_cells=total-len(self.last_seen), covered_fraction=len(self.last_seen)/total,
                    new_cells=self.new_cells, relative_heights_m=self.relative_heights_m,
                    basis="observed_cone_whole_cell_height_interval")

class SurveySearchRoute(StripSearchRoute):
    def __init__(self, bounds, partition_index=0, partition_count=3,
                 lane_spacing_m=700.0, arrival_radius_m=60.0, grid_size_m=100.0):
        super().__init__(bounds, partition_index, partition_count, lane_spacing_m,
                         arrival_radius_m, grid_size_m)
        self.coverage = SearchCoverageGrid(self.partition, grid_size_m)
        self.north_south = self.coverage.height >= self.coverage.width
        self.phase = "SURVEY"
        self.initial_heading = 0.0
        self.actual_spacing_m = None
        self.supplement_legs = 0
        self.survey_completed = False
        self.rejoin_lookahead_m = 250.0
        self._max_leg_progress_m = 0.0
        self._resume_pending = False
        self._rejoin_target = None
        self._rejoin_progress_m = None

    def _point(self, along, across):
        return self.coverage.point(across, along) if self.north_south else self.coverage.point(along, across)

    def _build(self, position):
        g = self.coverage
        length, width = (g.height,g.width) if self.north_south else (g.width,g.height)
        count = max(1, math.ceil(width/self.lane_spacing_m))
        self.actual_spacing_m = width/count
        lanes = [(i+.5)*self.actual_spacing_m for i in range(count)]
        x,y = g.xy(position)
        along, across = (y,x) if self.north_south else (x,y)
        lo, hi = min(50.,length/4), max(length-50.,length*3/4)
        entry = max(lo,min(hi,along))
        first = min(lanes,key=lambda c:abs(c-across))
        component = math.cos(math.radians(self.initial_heading)) if self.north_south else math.sin(math.radians(self.initial_heading))
        end,other = (hi,lo) if component >= 0 else (lo,hi)
        # 就近进入长航段，未飞过的首条半段保留到本轮最后，避免先绕远到角落。
        self.waypoints = [self._point(entry,first),self._point(end,first)]
        remaining = [c for c in lanes if c != first]
        while remaining:
            candidates = [(self._distance(self.waypoints[-1],self._point(a,c)),c,a,b)
                          for c in remaining for a,b in ((lo,hi),(hi,lo))]
            _,c,a,b = min(candidates)
            self.waypoints.extend((self._point(a,c),self._point(b,c)))
            remaining.remove(c)
        if abs(entry-other) > self.arrival_radius_m:
            self.waypoints.extend((self._point(other,first),self._point(entry,first)))

    def _next_supplement(self, position):
        g = self.coverage
        across_n,along_n = (g.nx,g.ny) if self.north_south else (g.ny,g.nx)
        across_step,along_step = (g.dx,g.dy) if self.north_south else (g.dy,g.dx)
        runs = []
        for col in range(across_n):
            start = None
            for row in range(along_n+1):
                cell = (col,row) if self.north_south else (row,col)
                missing = row < along_n and cell not in g.last_seen
                if missing and start is None:
                    start = row
                if not missing and start is not None:
                    # 将同列连续空白组成一段；选择后保持固定，直到到达再选下一段。
                    # 两端留出到点半径，确保实际穿过空白，而非尚未观察就反复到点。
                    padding=self.arrival_radius_m+along_step/2
                    a=self._point(max(0.,(start+.5)*along_step-padding),(col+.5)*across_step)
                    b=self._point(min(along_n*along_step,(row-.5)*along_step+padding),(col+.5)*across_step)
                    runs.extend(((self._distance(position,a),a,b),(self._distance(position,b),b,a)))
                    start=None
        if runs:
            _,a,b=min(runs)
            self.phase="SUPPLEMENT"
        else:
            # 已覆盖只代表历史观察过，后续优先重访最久未观察的网格。
            cell=min(g.last_seen,key=g.last_seen.get)
            along,across=(cell[1],cell[0]) if self.north_south else (cell[0],cell[1])
            padding=self.arrival_radius_m+along_step/2
            a=self._point(max(0.,(along+.5)*along_step-padding),(across+.5)*across_step)
            b=self._point(min(along_n*along_step,(along+.5)*along_step+padding),(across+.5)*across_step)
            if self._distance(position,b)<self._distance(position,a):
                a,b=b,a
            self.phase="REVISIT"
        self.waypoints=[a,b] if a!=b else [a]
        self.index=0
        self._max_leg_progress_m=0.0
        self._rejoin_target=None
        self._rejoin_progress_m=None
        self.supplement_legs+=1

    def pause(self):
        if self._active:
            self._resume_pending = True
            self._rejoin_target = None
            self._rejoin_progress_m = None
        self._active = False

    def _leg_projection(self, position):
        if self.index == 0:
            return None
        start, end = self.waypoints[self.index - 1:self.index + 1]
        east = (end[1] - start[1]) * self._lon_scale
        north = (end[0] - start[0]) * 111320.0
        length = math.hypot(east, north)
        if length <= 1e-6:
            return None
        own_east = (position[1] - start[1]) * self._lon_scale
        own_north = (position[0] - start[0]) * 111320.0
        progress = max(0.0, min(length, (own_east * east + own_north * north) / length))
        return progress, length, start, end

    def _advance(self, position):
        self.completed_waypoints += 1
        self.index += 1
        self._max_leg_progress_m = 0.0
        self._rejoin_target = None
        self._rejoin_progress_m = None
        if self.index == len(self.waypoints):
            if self.phase == "SURVEY":
                self.survey_completed = True
                self.completed_passes += 1
            self._next_supplement(position)

    def target(self, position, heading=0.0):
        if not self.waypoints:
            self.initial_heading=heading
            self._build(position)
        self._active=True
        if self._resume_pending:
            self._resume_pending = False
            leg = self._leg_projection(position)
            if leg is not None:
                projected, length, start, end = leg
                progress = max(projected, self._max_leg_progress_m)
                self._max_leg_progress_m = progress
                if progress >= length:
                    self._advance(position)
                else:
                    rejoin_progress = min(progress + self.rejoin_lookahead_m, length)
                    if rejoin_progress < length:
                        fraction = rejoin_progress / length
                        self._rejoin_target = (
                            start[0] + (end[0] - start[0]) * fraction,
                            start[1] + (end[1] - start[1]) * fraction)
                        self._rejoin_progress_m = rejoin_progress
        leg = self._leg_projection(position)
        if leg is not None:
            self._max_leg_progress_m = max(self._max_leg_progress_m, leg[0])
        if self._rejoin_target is not None:
            if (self._distance(position, self._rejoin_target) > self.arrival_radius_m
                    and (leg is None or leg[0] < self._rejoin_progress_m)):
                return self._rejoin_target
            self._rejoin_target = None
            self._rejoin_progress_m = None
        if self._distance(position,self.waypoints[self.index])<=self.arrival_radius_m:
            self._advance(position)
        return self.waypoints[self.index]

    def clamp_position(self, position):
        return tuple(max(self.bounds[0][i],min(self.bounds[1][i],position[i])) for i in (0,1))

    @property
    def summary(self):
        return dict(super().summary, phase=self.phase, axis="NS" if self.north_south else "EW",
                    actual_spacing_m=self.actual_spacing_m, survey_completed=self.survey_completed,
                    supplement_legs=self.supplement_legs, coverage=self.coverage.summary)




# ===== V4 来源：coordinated_search.py =====
class CoordinatedSweepRoute:
    def __init__(self, bounds, uid, members, lane_spacing_m=700.0,
                 arrival_radius_m=60.0, grid_size_m=100.0):
        self.bounds = bounds
        self.uid = str(uid)
        self.members = tuple(str(member) for member in members)
        self.lane_spacing_m = lane_spacing_m
        self.arrival_radius_m = arrival_radius_m
        self.coverage = SearchCoverageGrid(bounds, grid_size_m)
        (south, west), (north, east) = bounds
        self.lon_scale = 111320.0 * math.cos(math.radians((south + north) / 2.0))
        ew_length = (east - west) * self.lon_scale
        ns_length = (north - south) * 111320.0
        self.short_axis = "EW" if ew_length <= ns_length else "NS"
        self.short_length_m = min(ew_length, ns_length)
        self.long_length_m = max(ew_length, ns_length)
        self.sector_bounds = tuple((i * self.short_length_m / 3.0,
                                    (i + 1) * self.short_length_m / 3.0)
                                   for i in range(3))

        self.initialized = False
        self.mode = "WAIT_PEERS"
        self.sector_by_uid = {}
        self.home_sector = None
        self.active_sector_low = None
        self.active_sector_high = None
        self.align_v = None
        self.frontier_v = None
        self.cross_direction = 1
        self.advance_direction = 1
        self._align_direction = 1
        self._resume_pending = False
        self.takeover_master_uid = None
        self.takeover_follower_uid = None
        self.pair_midpoint = None
        self.last_target = None
        self.turn_count = 0
        self._events = []
        self.last_search_position = None

    def _uv(self, position):
        (south, west), _ = self.bounds
        east = (position[1] - west) * self.lon_scale
        north = (position[0] - south) * 111320.0
        return (east, north) if self.short_axis == "EW" else (north, east)

    def _position(self, u, v):
        (south, west), _ = self.bounds
        if self.short_axis == "EW":
            return south + v / 111320.0, west + u / self.lon_scale
        return south + u / 111320.0, west + v / self.lon_scale

    def _fresh_positions(self, position, now, peers):
        positions = {self.uid: position}
        for uid in self.members:
            if uid == self.uid or uid not in peers:
                continue
            peer_position, seen, _ = peers[uid]
            if now - seen <= 5.0:
                positions[uid] = peer_position
        return positions

    def _set_mode(self, mode):
        if mode == self.mode:
            return
        previous = self.mode
        self.mode = mode
        self._events.append(("search_mode_changed", {
            "previous": previous,
            "current": mode,
            "master_uid": self.takeover_master_uid,
            "follower_uid": self.takeover_follower_uid,
            "active_sector": [self.active_sector_low, self.active_sector_high],
            "frontier_v": self.frontier_v,
        }))

    def _initialize(self, positions):
        local = {uid: self._uv(positions[uid]) for uid in self.members}
        ordered = sorted(self.members, key=lambda uid: (local[uid][0], uid))
        self.sector_by_uid = {uid: sector for sector, uid in enumerate(ordered)}
        self.home_sector = self.sector_by_uid[self.uid]
        self.active_sector_low = self.home_sector
        self.active_sector_high = self.home_sector
        self.align_v = sorted(item[1] for item in local.values())[1]
        self.frontier_v = self.align_v
        middle_u = local[ordered[1]][0]
        low, high = self.sector_bounds[1]
        self.cross_direction = -1 if middle_u - low <= high - middle_u else 1
        self._align_direction = self.cross_direction
        self.initialized = True
        self._events.append(("search_plan_initialized", {
            "short_axis": self.short_axis,
            "map_short_length_m": self.short_length_m,
            "map_long_length_m": self.long_length_m,
            "sector_by_uid": dict(self.sector_by_uid),
            "align_v": self.align_v,
            "cross_direction": self.cross_direction,
            "advance_direction": self.advance_direction,
        }))
        self._set_mode("ALIGN")

    def _active_bounds(self):
        return (self.sector_bounds[self.active_sector_low][0],
                self.sector_bounds[self.active_sector_high][1])

    def _pair(self, now, peers):
        masters = []
        followers = []
        for uid in self.members:
            if uid == self.uid or uid not in peers or uid not in self.sector_by_uid:
                continue
            position, seen, state = peers[uid]
            if now - seen > 5.0:
                continue
            if state in ("CALLING", "COOP_TRACK_M"):
                masters.append((uid, position))
            elif state in ("FOLLOWER_APPROACH", "COOP_TRACK_F"):
                followers.append((uid, position))
        if len(masters) != 1 or len(followers) != 1:
            return None
        master_uid, master_position = masters[0]
        follower_uid, follower_position = followers[0]
        master_sector = self.sector_by_uid[master_uid]
        follower_sector = self.sector_by_uid[follower_uid]
        if ((master_sector == 0 and follower_sector == 1 and self.home_sector == 2)
                or (master_sector == 2 and follower_sector == 1 and self.home_sector == 0)):
            return master_uid, follower_uid, master_position, follower_position
        return None

    def _takeover(self, pair):
        master_uid, follower_uid, master_position, follower_position = pair
        self.takeover_master_uid = master_uid
        self.takeover_follower_uid = follower_uid
        self.active_sector_low = min(self.home_sector, 1)
        self.active_sector_high = max(self.home_sector, 1)
        u1, v1 = self._uv(master_position)
        u2, v2 = self._uv(follower_position)
        self.pair_midpoint = self._position((u1 + u2) / 2.0, (v1 + v2) / 2.0)
        # 双机中点可沿长轴前进或倒退，不做滤波和单调约束。
        self.frontier_v = (v1 + v2) / 2.0
        self._set_mode("TAKEOVER")

    def _restore_sweep(self, own_v):
        self.takeover_master_uid = None
        self.takeover_follower_uid = None
        self.pair_midpoint = None
        self.active_sector_low = self.home_sector
        self.active_sector_high = self.home_sector
        self.frontier_v = own_v
        self._set_mode("SWEEP")

    def _advance_frontier(self):
        next_v = self.frontier_v + self.advance_direction * self.lane_spacing_m
        if next_v >= self.long_length_m:
            self.frontier_v = self.long_length_m
            self.advance_direction = -1
        elif next_v <= 0.0:
            self.frontier_v = 0.0
            self.advance_direction = 1
        else:
            self.frontier_v = next_v

    def target(self, position, now, peers):
        positions = self._fresh_positions(position, now, peers)
        newly_initialized = False
        if not self.initialized:
            if len(positions) != len(self.members):
                self.last_target = None
                return None
            self._initialize(positions)
            newly_initialized = True
        own_u, own_v = self._uv(position)
        resumed = self._resume_pending and not newly_initialized
        self._resume_pending = False
        if resumed:
            self._restore_sweep(own_v)

        if self.mode == "ALIGN":
            aligned = (abs(own_v - self.align_v) <= self.arrival_radius_m
                       and len(positions) == len(self.members)
                       and all(abs(self._uv(peer_position)[1] - self.align_v)
                               <= self.arrival_radius_m for peer_position in positions.values()))
            if aligned:
                self._set_mode("SWEEP")
            elif abs(own_v - self.align_v) > self.arrival_radius_m:
                self.last_target = self._position(own_u, self.align_v)
                return self.last_target
            else:
                low, high = self._active_bounds()
                end_u = high if self._align_direction > 0 else low
                if abs(own_u - end_u) <= self.arrival_radius_m:
                    self._align_direction *= -1
                    end_u = high if self._align_direction > 0 else low
                self.last_target = self._position(end_u, self.align_v)
                return self.last_target

        pair = None if resumed else self._pair(now, peers)
        if pair is not None:
            self._takeover(pair)
        elif self.mode == "TAKEOVER":
            self._restore_sweep(own_v)

        low, high = self._active_bounds()
        end_u = high if self.cross_direction > 0 else low
        if math.hypot(own_u - end_u, own_v - self.frontier_v) <= self.arrival_radius_m:
            self.cross_direction *= -1
            self.turn_count += 1
            if self.mode == "SWEEP":
                self._advance_frontier()
            end_u = high if self.cross_direction > 0 else low
        self.last_target = self._position(end_u, self.frontier_v)
        return self.last_target

    def observe_search_position(self, position):
        self.last_search_position = position

    def pause(self):
        self._resume_pending = True

    def preferred_partner_uids(self):
        if not self.initialized:
            return None
        sectors = (1,) if self.home_sector in (0, 2) else (0, 2)
        return tuple(uid for uid in self.members if self.sector_by_uid[uid] in sectors)

    def drain_events(self):
        events = self._events
        self._events = []
        return events

    @property
    def trace_state(self):
        return {
            "mode": self.mode,
            "short_axis": self.short_axis,
            "home_sector": self.home_sector,
            "active_sector": ([self.active_sector_low, self.active_sector_high]
                              if self.initialized else None),
            "frontier_v_m": self.frontier_v,
            "cross_direction": self.cross_direction,
            "advance_direction": self.advance_direction,
            "takeover_master_uid": self.takeover_master_uid,
            "takeover_follower_uid": self.takeover_follower_uid,
            "pair_midpoint": self.pair_midpoint,
            "target": self.last_target,
        }

    @property
    def summary(self):
        return {
            "short_axis": self.short_axis,
            "home_sector": self.home_sector,
            "mode": self.mode,
            "turn_count": self.turn_count,
            "lane_spacing_m": self.lane_spacing_m,
            "coverage": {"covered_fraction": self.coverage.summary["covered_fraction"]},
        }




# ===== V4 来源：search_gimbal.py =====
def _wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0

@dataclass(frozen=True)
class SearchGimbalConfig:
    side_angle_deg: float = 35.0
    side_dwell_s: float = 0.8
    center_dwell_s: float = 0.4
    tolerance_deg: float = 3.0
    near_nadir_m: float = 80.0

class SearchGimbalController:
    def __init__(self, config=None):
        self.config = config or SearchGimbalConfig()
        self.mode = "IDLE"
        self.step_index = 0
        self.arrived_since = None
        self.cycles = 0
        self.holds = 0
        self.hold_tilt = None
        self.hold_position = None
        self.command = (0.0, -90.0)

    def suspend(self):
        """离开搜索后停止巡视，协同云台仍由原控制流程负责。"""
        self.mode = "IDLE"
        self.arrived_since = None
        self.hold_tilt = None
        self.hold_position = None

    def scan(self, now, route_heading, heading, pan, tilt):
        c = self.config
        # 左右换边前先回正下方，再转水平轴，避免把换边误当成瞬时横扫。
        steps = ((-90, -90, 0), (-90, -90 + c.side_angle_deg, c.side_dwell_s),
                 (-90, -90, c.center_dwell_s), (90, -90, 0),
                 (90, -90 + c.side_angle_deg, c.side_dwell_s),
                 (90, -90, c.center_dwell_s))
        if self.mode != "SCAN":
            self.step_index = 0
            self.arrived_since = None
            self.hold_tilt = None
            self.hold_position = None
        self.mode = "SCAN"
        offset, target_tilt, dwell = steps[self.step_index]
        target_pan = _wrap(route_heading + offset - heading)
        arrived = (abs(_wrap(pan - target_pan)) <= c.tolerance_deg
                   and abs(tilt - target_tilt) <= c.tolerance_deg)
        if not arrived:
            self.arrived_since = None
        else:
            if self.arrived_since is None:
                self.arrived_since = now
            if now - self.arrived_since >= dwell:
                self.step_index = (self.step_index + 1) % len(steps)
                self.cycles += int(self.step_index == 0)
                self.arrived_since = None
                offset, target_tilt, _ = steps[self.step_index]
                target_pan = _wrap(route_heading + offset - heading)
        self.command = (target_pan, target_tilt)
        return self.command

    def observe_candidate(self, position, candidate, heading, actual_tilt):
        """保持发现时俯仰并修正方位；靠近后回正下方，不反算目标高度。"""
        switched = (self.hold_position is not None
                    and _haversine_m(*self.hold_position, *candidate) > 80.0)
        if self.mode != "OBSERVE" or switched:
            self.hold_tilt = actual_tilt
            self.holds += 1
        self.mode = "OBSERVE"
        self.arrived_since = None
        self.hold_position = candidate
        pan = _wrap(_bearing_deg(*position, *candidate) - heading)
        tilt = self.hold_tilt
        if _haversine_m(*position, *candidate) < self.config.near_nadir_m:
            pan, tilt = 0.0, -90.0
        self.command = (pan, tilt)
        return self.command

    @property
    def summary(self):
        return dict(mode=self.mode, step=self.step_index, cycles=self.cycles,
                    holds=self.holds, arrived_since_s=self.arrived_since,
                    side_angle_deg=self.config.side_angle_deg,
                    command_pan_deg=self.command[0], command_tilt_deg=self.command[1],
                    hold_tilt_deg=self.hold_tilt)

class GimbalHandoffController:
    """将已观察到的目标交给协同控制，不估计目标高程或替代有效锁定判定。"""

    def __init__(self, near_nadir_m=80.0, lower_rate_dps=6.0):
        self.near_nadir_m = near_nadir_m
        self.lower_rate_dps = lower_rate_dps
        self.reset()

    def reset(self):
        self.key = None
        self.mode = "IDLE"
        self.tilt = None

    def update(self, key, dt, position, target, heading, actual_pan, actual_tilt,
               target_visible):
        if key != self.key:
            self.reset()
            self.key = key
        if self.mode == "IDLE":
            # 从机仅收到召唤坐标时，不继承搜索中另一辆车的云台姿态。
            if target is None or not target_visible:
                return 0.0, -90.0
            self.tilt = actual_tilt
            self.mode = "HOLD"
        if self.mode == "NADIR":
            return 0.0, -90.0
        if target is None:
            return actual_pan, actual_tilt
        pan = _wrap(_bearing_deg(*position, *target) - heading)
        distance = _haversine_m(*position, *target)
        if not target_visible:
            # 缺少新观测时保持实际姿态，不继续把目标推向视野边缘。
            self.tilt = actual_tilt
            self.mode = "HOLD"
        elif distance <= self.near_nadir_m:
            self.mode = "LOWERING"
            # 限制单次步长，避免长决策间隔再次形成瞬时朝下跳变。
            step = self.lower_rate_dps * min(max(dt, 0.0), 0.25)
            self.tilt = max(-90.0, actual_tilt - step)
            if actual_tilt <= -89.9:
                self.mode = "NADIR"
                self.tilt = -90.0
                return 0.0, -90.0
        else:
            self.mode = "HOLD"
            self.tilt = actual_tilt
        return pan, self.tilt

    @property
    def summary(self):
        return dict(mode=self.mode, key=self.key, tilt_deg=self.tilt,
                    near_nadir_m=self.near_nadir_m, lower_rate_dps=self.lower_rate_dps)




# ===== V4 来源：v3_simple_control.py =====
EARTH_RADIUS_M = 6_371_000.0

def _wrap_signed_deg(angle_deg):
    """将相对方位约束到云台 pan 使用的 [-180, 180) 度。"""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0

def ground_distance_m(first_position, second_position):
    """按球面海佛距离计算两个经纬度点的水平距离。"""
    lat1, lon1 = map(float, first_position)
    lat2, lon2 = map(float, second_position)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lon = math.radians(lon2 - lon1)
    value = (math.sin(delta_phi / 2.0) ** 2
             + math.cos(phi1) * math.cos(phi2)
             * math.sin(delta_lon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(
        max(0.0, min(1.0, value))))

def bearing_deg(first_position, second_position):
    """返回从第一个经纬度点指向第二个点的真方位角。"""
    lat1, lon1 = map(float, first_position)
    lat2, lon2 = map(float, second_position)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    east = math.sin(delta_lon) * math.cos(phi2)
    north = (math.cos(phi1) * math.sin(phi2)
             - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon))
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0

def offset_position(origin, east_m, north_m):
    """把局部东、北米偏移换成附近的经纬度航点。"""
    lat, lon = map(float, origin)
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) <= 1e-6:
        return lat, lon
    return (
        lat + float(north_m) / 111_320.0,
        lon + float(east_m) / (111_320.0 * cos_lat),
    )

@dataclass(frozen=True)
class GroundAim:
    """指向一个已知经纬高坐标的云台指令解。"""

    target_position: tuple[float, float]
    target_alt_m: float
    ground_distance_m: float
    bearing_deg: float
    pan_deg: float
    tilt_deg: float

def solve_ground_aim(self_position, self_alt_m, self_heading_deg,
                     target_position, *, target_alt_m=0.0):
    """用本拍飞机位姿反解机体相对 pan 与世界俯仰 tilt。

    heading 和方位都以正北为零度、顺时针为正；pan 是相对机头
    的有符号夹角；tilt 向上为正。H=0 时目标在 500 米无人机下方，
    因而 tilt 为负值。
    """
    own = tuple(map(float, self_position))
    target = tuple(map(float, target_position))
    horizontal = ground_distance_m(own, target)
    target_bearing = bearing_deg(own, target)
    height_delta = float(target_alt_m) - float(self_alt_m)
    if horizontal <= 1e-6:
        # 正上方/正下方时方位无定义，pan 回中避免每拍跳变。
        pan = 0.0
        tilt = 90.0 if height_delta > 0.0 else -90.0
    else:
        pan = _wrap_signed_deg(target_bearing - float(self_heading_deg))
        tilt = math.degrees(math.atan2(height_delta, horizontal))
    return GroundAim(
        target_position=target,
        target_alt_m=float(target_alt_m),
        ground_distance_m=horizontal,
        bearing_deg=target_bearing,
        pan_deg=pan,
        tilt_deg=tilt,
    )

@dataclass(frozen=True)
class SimpleCoopControlConfig:
    """005 开发记录中的简化从机控制门槛。"""

    master_gate_m: float = 220.0
    target_gate_m: float = 200.0
    target_alt_m: float = 0.0
    follower_speed_mps: float = 22.0
    follower_loiter_radius_m: float = 100.0
    parallax_sample_baseline_m: float = 45.0
    parallax_sample_speed_mps: float = 22.0
    coop_orbit_radius_m: float = 130.0
    coop_orbit_period_s: float = 48.0

@dataclass(frozen=True)
class FollowerGuidance:
    """FOLLOWER 每个控制 tick 的不可变导引快照。"""

    fly_to_position: tuple[float, float] | None
    fly_to_speed_mps: float | None
    fly_to_loiter_radius_m: float | None
    uav_distance_to_master_m: float | None
    target_distance_m: float | None
    master_gate_m: float
    target_gate_m: float
    within_master_gate: bool
    within_target_gate: bool
    rendezvous_ready: bool
    guidance_enabled: bool
    aiming_enabled: bool
    aim_target_lat: float | None
    aim_target_lon: float | None
    gimbal_pan_cmd_deg: float | None
    gimbal_tilt_cmd_deg: float | None

    def as_evidence(self):
        """返回稳定字段名，供旁路记录逐拍审计门槛与瞄准。"""
        return asdict(self)

    @property
    def ready(self):
        """为接入层提供简短的双门槛别名。"""
        return self.rendezvous_ready

@dataclass(frozen=True)
class OrbitGuidance:
    """围绕共享目标的受控双螺旋航点。"""

    fly_to_position: tuple[float, float]
    planned_separation_m: float
    radius_m: float
    phase_deg: float

class SimpleCoopControl:
    """FOLLOWER 赶赴 MASTER 目标坐标、到位后直指 H=0 坐标。"""

    def __init__(self, config=None):
        self.config = config or SimpleCoopControlConfig()
        self.reset()

    def reset(self):
        """离开协同会话时清除从机已到位锁存。"""
        self._follower_session_key = None
        self._follower_aim_latched = False

    def master_aim(self, *, self_position, self_alt_m, self_heading_deg,
                   target_position):
        """MASTER 每拍用本机实时位姿重算对自己目标的指向。"""
        if target_position is None:
            return None
        return solve_ground_aim(
            self_position,
            self_alt_m,
            self_heading_deg,
            target_position,
            target_alt_m=self.config.target_alt_m,
        )

    def parallax_sample(self, *, self_position, target_position):
        """生成垂直当前视线的约四十五米采样航点。"""
        if target_position is None:
            return None
        bearing = bearing_deg(self_position, target_position)
        lateral = math.radians(bearing + 90.0)
        distance = self.config.parallax_sample_baseline_m
        return offset_position(
            self_position,
            distance * math.sin(lateral),
            distance * math.cos(lateral),
        )

    def dual_orbit(self, *, target_position, now_s, role):
        """返回 MASTER/FOLLOWER 反相的目标周围航点。

        半径一百三十米时理想相对间距为二百六十米，高于二百二十米约束；
        执行器仍须从当前实际位置追赶该航点，因此证据中同时保留计划间距。
        """
        if target_position is None:
            return None
        phase = math.tau * (float(now_s) / self.config.coop_orbit_period_s)
        if str(role) == "FOLLOWER":
            phase += math.pi
        radius = self.config.coop_orbit_radius_m
        return OrbitGuidance(
            fly_to_position=offset_position(
                target_position, radius * math.sin(phase), radius * math.cos(phase)),
            planned_separation_m=2.0 * radius,
            radius_m=radius,
            phase_deg=math.degrees(phase) % 360.0,
        )

    def follower(self, *, self_position, self_alt_m, self_heading_deg,
                 master_position, follow_position, session_key=None):
        """按 MASTER 机位与广播目标坐标生成从机导引。

        follow_position 一旦可用就持续作为 fly_to 目的地；仅当本机
        到 MASTER 不超过 220 米且到该目标估计不超过 200 米时，
        才锁存并返回 point_gimbal 所需的 pan/tilt。同一会话锁存后即使
        短时离开门槛也仍持续指向；会话改变或 reset 后重新判定。
        """
        if session_key != self._follower_session_key:
            self._follower_session_key = session_key
            self._follower_aim_latched = False
        if follow_position is None:
            return FollowerGuidance(
                fly_to_position=None,
                fly_to_speed_mps=None,
                fly_to_loiter_radius_m=None,
                uav_distance_to_master_m=(
                    None if master_position is None
                    else ground_distance_m(self_position, master_position)
                ),
                target_distance_m=None,
                master_gate_m=self.config.master_gate_m,
                target_gate_m=self.config.target_gate_m,
                within_master_gate=False,
                within_target_gate=False,
                rendezvous_ready=False,
                guidance_enabled=False,
                aiming_enabled=False,
                aim_target_lat=None,
                aim_target_lon=None,
                gimbal_pan_cmd_deg=None,
                gimbal_tilt_cmd_deg=None,
            )

        target = tuple(map(float, follow_position))
        distance_to_target = ground_distance_m(self_position, target)
        distance_to_master = (
            None if master_position is None
            else ground_distance_m(self_position, master_position)
        )
        within_master = (distance_to_master is not None
                         and distance_to_master <= self.config.master_gate_m)
        within_target = distance_to_target <= self.config.target_gate_m
        ready = within_master and within_target
        self._follower_aim_latched = self._follower_aim_latched or ready
        aim = (solve_ground_aim(
            self_position,
            self_alt_m,
            self_heading_deg,
            target,
            target_alt_m=self.config.target_alt_m,
        ) if self._follower_aim_latched else None)
        return FollowerGuidance(
            fly_to_position=target,
            fly_to_speed_mps=self.config.follower_speed_mps,
            fly_to_loiter_radius_m=self.config.follower_loiter_radius_m,
            uav_distance_to_master_m=distance_to_master,
            target_distance_m=distance_to_target,
            master_gate_m=self.config.master_gate_m,
            target_gate_m=self.config.target_gate_m,
            within_master_gate=within_master,
            within_target_gate=within_target,
            rendezvous_ready=ready,
            guidance_enabled=True,
            aiming_enabled=self._follower_aim_latched,
            aim_target_lat=target[0] if self._follower_aim_latched else None,
            aim_target_lon=target[1] if self._follower_aim_latched else None,
            gimbal_pan_cmd_deg=aim.pan_deg if aim is not None else None,
            gimbal_tilt_cmd_deg=aim.tilt_deg if aim is not None else None,
        )




# ===== V4 来源：local_motion_flow.py =====
def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

def _box(value: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    raw = _field(value, "bbox_xyxy", _field(value, "xyxy"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return None
    try:
        box = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if not (all(math.isfinite(v) for v in box) and 0 <= x1 < x2 <= width
            and 0 <= y1 < y2 <= height):
        return None
    return box  # type: ignore[return-value]

def _anchor(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.array(((box[0] + box[2]) * 0.5, box[3]), dtype=np.float64)

@dataclass(frozen=True)
class LocalMotionParameters:
    """第一版阈值；像素与框高度同时约束判定。"""

    max_corners: int = 600
    fb_error_px: float = 1.5
    min_local_points: int = 10
    min_inliers: int = 8
    min_inlier_ratio: float = 0.55
    max_rmse_px: float = 2.0
    window_transitions: int = 5
    max_transition_gap_s: float = 1.5

@dataclass
class _TrackWindow:
    transitions: deque = field(default_factory=lambda: deque(maxlen=8))
    last_raw: str = "UNKNOWN"

class LocalMotionDetector:
    """每架 UAV 持有一个实例；调用方按图像采集顺序提交新帧。"""

    def __init__(self, parameters: LocalMotionParameters | None = None) -> None:
        self.parameters = parameters or LocalMotionParameters()
        self.reset()

    def reset(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._prev_boxes: dict[int, tuple[float, float, float, float]] = {}
        self._prev_all_boxes: tuple[tuple[float, float, float, float], ...] = ()
        self._prev_frame_id: str | None = None
        self._prev_time: float | None = None
        self._tracks: dict[int, _TrackWindow] = {}

    @staticmethod
    def _unknown(reason: str, **evidence: Any) -> dict[str, Any]:
        return {"decision": "UNKNOWN", "raw_decision": "UNKNOWN",
                "confirmed_decision": "UNKNOWN", "reason": reason,
                "local_feature_count": 0, "affine_inlier_count": 0,
                "affine_inlier_ratio": None, "affine_rmse_px": None,
                "window_valid_frames": 0, "static_predicted_pixel": None,
                "actual_bottom_center": None, "endpoint_error_px": None,
                "bbox_height_px": None, "normalized_error": None,
                "T_static": None, "T_moving": None, **evidence}

    @staticmethod
    def _detect_boxes(detections: Sequence[Any], width: int, height: int) -> dict[int, tuple[float, float, float, float]]:
        boxes = {}
        for detection in detections:
            track_id = _field(detection, "track_id")
            box = _box(detection, width, height)
            try:
                key = int(track_id)
            except (TypeError, ValueError):
                continue
            if key < 0 or box is None:
                continue
            boxes[key] = box
        return boxes

    @staticmethod
    def _local_roi(
        box: tuple[float, float, float, float], image_width: int, image_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx = (x1 + x2) * 0.5
        half_w = max(2.0 * bw, 50.0)
        bottom_extra = max(1.5 * bh, 45.0)
        xa = max(0, int(math.floor(cx - half_w)))
        xb = min(image_width, int(math.ceil(cx + half_w)))
        ya = max(0, int(math.floor(y1 + 0.3 * bh)))
        yb = min(image_height, int(math.ceil(y2 + bottom_extra)))
        return xa, ya, xb, yb

    @staticmethod
    def _background_mask(
        shape: tuple[int, int],
        boxes: Sequence[tuple[float, float, float, float]],
        focus_boxes: Sequence[tuple[float, float, float, float]] = (),
    ) -> np.ndarray:
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8) if focus_boxes else np.full(
            (height, width), 255, dtype=np.uint8)
        # 有候选目标时把角点预算集中到其脚下及两侧。
        for box in focus_boxes:
            xa, ya, xb, yb = LocalMotionDetector._local_roi(box, width, height)
            mask[ya:yb, xa:xb] = 255
        for x1, y1, x2, y2 in boxes:
            dx, dy = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
            xa, ya = max(0, int(math.floor(x1 - dx))), max(0, int(math.floor(y1 - dy)))
            xb, yb = min(width, int(math.ceil(x2 + dx))), min(height, int(math.ceil(y2 + dy)))
            mask[ya:yb, xa:xb] = 0
        return mask

    def _background_flow(
        self,
        gray: np.ndarray,
        all_boxes: tuple[tuple[float, float, float, float], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._prev_gray is not None
        previous_mask = self._background_mask(
            gray.shape, self._prev_all_boxes, tuple(self._prev_boxes.values()))
        corners = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=self.parameters.max_corners,
            qualityLevel=0.01, minDistance=7, blockSize=7, mask=previous_mask,
        )
        empty = np.empty((0, 2), dtype=np.float32)
        if corners is None:
            return empty, empty
        forward, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, corners, None, winSize=(31, 31), maxLevel=4,
        )
        if forward is None or forward_ok is None:
            return empty, empty
        valid_forward = (forward_ok[:, 0] == 1) & np.isfinite(forward[:, 0]).all(axis=1)
        corners, forward = corners[valid_forward], forward[valid_forward]
        if len(corners) == 0:
            return empty, empty
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, forward, None, winSize=(31, 31), maxLevel=4,
        )
        if backward is None or backward_ok is None:
            return empty, empty
        old, new, back = corners[:, 0], forward[:, 0], backward[:, 0]
        good = ((backward_ok[:, 0] == 1)
                & np.isfinite(back).all(axis=1)
                & (np.linalg.norm(old - back, axis=1) < self.parameters.fb_error_px))
        # 当前帧的目标框也要排除，防止背景点落入移动车辆或遮挡处。
        current_mask = self._background_mask(gray.shape, all_boxes)
        x = np.clip(np.rint(new[:, 0]).astype(np.int32), 0, gray.shape[1] - 1)
        y = np.clip(np.rint(new[:, 1]).astype(np.int32), 0, gray.shape[0] - 1)
        good &= current_mask[y, x] != 0
        good &= ((new[:, 0] >= 0) & (new[:, 0] < gray.shape[1])
                 & (new[:, 1] >= 0) & (new[:, 1] < gray.shape[0]))
        return old[good], new[good]

    def _transition(
        self,
        track_id: int,
        old_box: tuple[float, float, float, float],
        new_box: tuple[float, float, float, float],
        old_points: np.ndarray,
        new_points: np.ndarray,
    ) -> dict[str, Any]:
        assert self._prev_gray is not None
        image_height, image_width = self._prev_gray.shape
        xa, ya, xb, yb = self._local_roi(old_box, image_width, image_height)
        local = ((old_points[:, 0] >= xa)
                 & (old_points[:, 0] < xb)
                 & (old_points[:, 1] >= ya)
                 & (old_points[:, 1] < yb))
        p0, p1 = old_points[local], new_points[local]
        count = len(p0)
        if count < self.parameters.min_local_points:
            return self._unknown("insufficient_local_features", local_feature_count=count)
        affine, inlier_mask = cv2.estimateAffine2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=1000, confidence=0.99, refineIters=10,
        )
        if affine is None or inlier_mask is None or not np.isfinite(affine).all():
            return self._unknown("affine_failed", local_feature_count=count)
        inliers = inlier_mask[:, 0].astype(bool)
        inlier_count = int(np.sum(inliers))
        ratio = inlier_count / count
        estimate = p0[inliers] @ affine[:, :2].T + affine[:, 2]
        rmse = float(np.sqrt(np.mean(np.sum((estimate - p1[inliers]) ** 2, axis=1)))) if inlier_count else math.inf
        evidence = {"local_feature_count": count, "affine_inlier_count": inlier_count,
                    "affine_inlier_ratio": ratio, "affine_rmse_px": rmse}
        if (inlier_count < self.parameters.min_inliers
                or ratio <= self.parameters.min_inlier_ratio
                or rmse >= self.parameters.max_rmse_px):
            return self._unknown("weak_local_affine", **evidence)

        window = self._tracks.setdefault(track_id, _TrackWindow(
            transitions=deque(maxlen=self.parameters.window_transitions)))
        window.transitions.append((affine.copy(), _anchor(old_box), _anchor(new_box),
                                   float(new_box[3] - new_box[1]), rmse))
        steps = tuple(window.transitions)
        evidence["window_valid_frames"] = len(steps)
        if len(steps) < self.parameters.window_transitions:
            window.last_raw = "UNKNOWN"
            return self._unknown("warmup", **evidence)
        predicted = steps[0][1].copy()
        previous_endpoint = None
        for matrix, old_anchor, new_anchor, _, _ in steps:
            # 坏帧没有仿射证据，用两侧已观测锚点补齐断层，不把该帧运动计入窗口。
            if previous_endpoint is not None:
                predicted += old_anchor - previous_endpoint
            predicted = matrix[:, :2] @ predicted + matrix[:, 2]
            previous_endpoint = new_anchor
        actual = steps[-1][2]
        endpoint_error = float(np.linalg.norm(actual - predicted))
        median_height = float(np.median([step[3] for step in steps]))
        sigma = float(np.median([step[4] for step in steps]))
        threshold_static = max(0.06 * median_height, 3.0, 2.0 * sigma)
        threshold_moving = max(0.18 * median_height, 6.0, 4.0 * sigma)
        raw = ("STATIC" if endpoint_error < threshold_static else
               "MOVING" if endpoint_error > threshold_moving else "UNKNOWN")
        confirmed = raw if raw != "UNKNOWN" and raw == window.last_raw else "UNKNOWN"
        window.last_raw = raw
        evidence.update({
            "static_predicted_pixel": predicted.tolist(),
            "actual_bottom_center": actual.tolist(),
            "endpoint_error_px": endpoint_error,
            "bbox_height_px": median_height,
            "normalized_error": endpoint_error / median_height,
            "T_static": threshold_static,
            "T_moving": threshold_moving,
        })
        return {"decision": confirmed, "raw_decision": raw,
                "confirmed_decision": confirmed,
                "reason": "confirmed" if confirmed != "UNKNOWN" else "awaiting_confirmation",
                **evidence}

    def update(
        self,
        frame_bgr: np.ndarray | None,
        detections: Sequence[Any],
        frame_time_s: float,
        *,
        frame_id: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        """返回按字符串 track_id 索引的证据；仅新且递增的同尺寸帧可累计。"""
        try:
            time_s = float(frame_time_s)
        except (TypeError, ValueError):
            self.reset()
            return {}
        if (frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or not math.isfinite(time_s)):
            self.reset()
            return {}
        frame_id = frame_id or str(time_s)
        height, width = frame_bgr.shape[:2]
        boxes = self._detect_boxes(detections, width, height)
        all_boxes = tuple(box for detection in detections
                          if (box := _box(detection, width, height)) is not None)
        if frame_id == self._prev_frame_id:
            return {tid: {"track_id": tid, **self._unknown("duplicate_frame")}
                    for tid in boxes}
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if (self._prev_gray is None or gray.shape != self._prev_gray.shape
                or self._prev_time is None or time_s <= self._prev_time
                or time_s - self._prev_time > self.parameters.max_transition_gap_s):
            self.reset()
            self._prev_gray, self._prev_boxes = gray, boxes
            self._prev_all_boxes = all_boxes
            self._prev_frame_id, self._prev_time = frame_id, time_s
            return {tid: {"track_id": tid, **self._unknown("first_or_discontinuous_frame")}
                    for tid in boxes}
        old_points, new_points = self._background_flow(gray, all_boxes)
        results = {}
        for tid, box in boxes.items():
            old_box = self._prev_boxes.get(tid)
            if old_box is None:
                self._tracks.pop(tid, None)
                results[tid] = {"track_id": tid, **self._unknown("new_track")}
                continue
            result = self._transition(tid, old_box, box, old_points, new_points)
            results[tid] = {"track_id": tid, **result}
        self._tracks = {tid: state for tid, state in self._tracks.items() if tid in boxes}
        self._prev_gray, self._prev_boxes = gray, boxes
        self._prev_all_boxes = all_boxes
        self._prev_frame_id, self._prev_time = frame_id, time_s
        return results




# ===== V4 来源：v4_entity.py =====
ENTITY_LOST_S = 5.0

def _center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

@dataclass
class Entity:
    entity_id: str
    bbox_xyxy: tuple[float, float, float, float] | None
    visible: bool
    missing_s: float
    lost: bool
    observed_frames: int
    last_box: tuple[float, float, float, float]
    last_seen_s: float

class EntityManager:
    def __init__(self, uid, lost_s=ENTITY_LOST_S):
        self.uid = str(uid)
        self.lost_s = float(lost_s)
        self.counter = 0
        self.current: Entity | None = None

    def clear(self):
        self.current = None

    def update(self, objects, image_size, now_s):
        """返回 (entity, event)；空观测保留已有实体直至超时。"""
        now_s = float(now_s)
        real = [item for item in objects if item.class_name == "real_vehicle"]
        entity = self.current
        if entity is None:
            if not real:
                return None, None
            width, height = image_size
            item = min(real, key=lambda obj: math.dist(_center(obj.bbox_xyxy),
                                                       (width * 0.5, height * 0.5)))
            self.counter += 1
            box = item.bbox_xyxy
            entity = Entity(f"uav_{self.uid}_entity_{self.counter}", box, True, 0.0,
                            False, 1, box, now_s)
            self.current = entity
            return entity, "entity_created"
        prediction = _center(entity.last_box)
        diagonal = math.hypot(entity.last_box[2] - entity.last_box[0],
                              entity.last_box[3] - entity.last_box[1])
        gate = max(220.0, 1.5 * diagonal)
        nearest = min(real, key=lambda obj: math.dist(_center(obj.bbox_xyxy), prediction),
                      default=None)
        if nearest is not None and math.dist(_center(nearest.bbox_xyxy), prediction) <= gate:
            entity.last_box = nearest.bbox_xyxy
            entity.last_seen_s = now_s
            entity.bbox_xyxy = nearest.bbox_xyxy
            entity.visible = True
            entity.missing_s = 0.0
            entity.observed_frames += 1
            return entity, "entity_matched"
        entity.bbox_xyxy = None
        entity.visible = False
        entity.missing_s = max(0.0, now_s - entity.last_seen_s)
        if entity.missing_s > self.lost_s:
            entity.lost = True
            self.current = None
            return entity, "entity_lost"
        return entity, "entity_missing"




# ===== V4 来源：v4_coordination.py =====
KINDS = {"H", "INVITE", "ACCEPT", "TARGET", "READY", "START", "DONE", "CANCEL"}

PERIODIC = {"H", "INVITE", "ACCEPT", "TARGET", "READY", "START"}

def _base36(value):
    value = int(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while value:
        result = digits[value % 36] + result
        value //= 36
    return sign + (result or "0")

def _from36(value):
    return int(value, 36)

def _position(position):
    return _base36(round(float(position[0]) * 1_000_000)), _base36(
        round(float(position[1]) * 1_000_000))

def _decode_position(lat, lon):
    return _from36(lat) / 1_000_000, _from36(lon) / 1_000_000

class V4Coordinator:
    def __init__(self, uid):
        self.uid = str(uid)
        self.peers = {}
        self.session = None
        self.master_uid = None
        self.partner_uid = None
        self.master_position = None
        self.target = None
        self.orbit_phase_deg = None
        self.orbit_start_s = None
        self.last_master_message_s = -1e9
        self.accepted = False
        self.ready = False
        self.started = False
        self.queue = deque()
        self.last_sent_s = -1e9
        self.last_queued = {}
        self.sent_events = 0
        self.received_events = 0
        self._seen_inbox = set()

    def clear_session(self):
        self.session = None
        self.master_uid = None
        self.partner_uid = None
        self.master_position = None
        self.target = None
        self.orbit_phase_deg = None
        self.orbit_start_s = None
        self.accepted = False
        self.ready = False
        self.started = False
        self.queue = deque(item for item in self.queue if item[0] in ("H", "DONE", "CANCEL"))

    def set_master(self, entity_id):
        self.clear_session()
        self.session = str(entity_id).removeprefix("uav_").replace("_entity_", ".")
        self.master_uid = self.uid

    def select_partner(self, own_position, now, allowed_uids=None):
        candidates = [(position, uid) for uid, (position, seen, state) in self.peers.items()
                      if now - seen <= 5.0 and state == "SEARCH"
                      and (allowed_uids is None or uid in allowed_uids)]
        if not candidates:
            return None
        return min(candidates, key=lambda item: ground_distance_m(own_position, item[0]))[1]

    def ingest(self, inbox, now, state):
        events = []
        for message in inbox:
            sender = str(getattr(message, "sender_uid", ""))
            payload = str(getattr(message, "payload", ""))
            if sender == self.uid:
                continue
            key = (sender, payload, getattr(message, "recv_time", None))
            if key in self._seen_inbox:
                continue
            self._seen_inbox.add(key)
            parts = payload.split("|")
            if len(parts) < 3 or parts[0] != "V4" or parts[1] not in KINDS:
                continue
            kind = parts[1]
            self.received_events += 1
            if kind == "H" and len(parts) == 5:
                try:
                    position = _decode_position(parts[2], parts[3])
                    self.peers[sender] = (position, now, parts[4])
                    if sender == self.master_uid:
                        self.master_position = position
                except ValueError:
                    pass
                continue
            session = parts[2]
            if (kind == "INVITE" and state == "SEARCH" and self.session is None
                    and len(parts) == 6 and parts[3] == self.uid):
                self.session = session
                self.master_uid = sender
                self.master_position = self.peers.get(sender, (None,))[0]
                self.last_master_message_s = now
                try:
                    self.target = _decode_position(parts[4], parts[5])
                except ValueError:
                    pass
                events.append("INVITE")
                continue
            if session != self.session:
                continue
            if self.master_uid == self.uid:
                if kind == "ACCEPT" and (self.partner_uid is None or self.partner_uid == sender):
                    self.partner_uid = sender
                    self.accepted = True
                    events.append("ACCEPT")
                elif kind == "READY" and sender == self.partner_uid:
                    self.ready = True
                    events.append("READY")
            elif sender == self.master_uid:
                self.last_master_message_s = now
                if kind == "TARGET" and len(parts) == 7:
                    try:
                        self.target = _decode_position(parts[3], parts[4])
                        self.master_position = _decode_position(parts[5], parts[6])
                        events.append("TARGET")
                    except ValueError:
                        pass
                elif kind == "START" and len(parts) == 5:
                    try:
                        self.orbit_phase_deg = _from36(parts[3]) / 10.0
                        self.orbit_start_s = _from36(parts[4]) / 1000.0
                    except ValueError:
                        continue
                    self.started = True
                    events.append("START")
                elif kind in ("DONE", "CANCEL"):
                    events.append(kind)
        return events

    def queue_message(self, kind, now, *, position=None, target=None, state=None,
                      phase_deg=None, start_s=None, period_s=0.0):
        if kind not in KINDS:
            raise ValueError(kind)
        if kind not in ("H",) and self.session is None:
            return
        if now - self.last_queued.get(kind, -1e9) < period_s:
            return
        self.last_queued[kind] = now
        parts = ["V4", kind]
        if kind == "H":
            if position is None:
                return
            parts.extend((*_position(position), str(state)))
        else:
            parts.append(self.session)
            if kind == "INVITE" and target is not None and self.partner_uid is not None:
                parts.extend((self.partner_uid, *_position(target)))
            elif kind == "TARGET" and target is not None and position is not None:
                parts.extend((*_position(target), *_position(position)))
            elif kind == "START" and phase_deg is not None and start_s is not None:
                parts.extend((_base36(round(float(phase_deg) * 10.0)),
                              _base36(round(float(start_s) * 1000.0))))
        payload = "|".join(parts)
        if len(payload.encode("utf-8")) > 50:
            raise ValueError("Agent4 通信载荷超过官方 50 字节上限")
        if kind in PERIODIC:
            self.queue = deque(item for item in self.queue if item[0] != kind)
        self.queue.append((kind, payload))

    def emit(self, now):
        if now - self.last_sent_s < 0.26 or not self.queue:
            return None, None
        priority = {"DONE": 0, "CANCEL": 0, "START": 1, "READY": 1,
                    "ACCEPT": 1, "INVITE": 2, "TARGET": 2, "H": 3}
        index = min(range(len(self.queue)), key=lambda i: (priority[self.queue[i][0]], i))
        kind, payload = self.queue[index]
        del self.queue[index]
        self.last_sent_s = now
        self.sent_events += 1
        return broadcast(payload), kind




# ===== V4 来源：v4_gimbal.py =====
DEAD_X = 45.0

DEAD_Y = 35.0

CONTROL_INTERVAL_S = 0.30

PAN_MAX_STEP_DEG = 6.5

TILT_MAX_STEP_DEG = 4.0

GAIN = 0.5

class VisualGimbal:
    def __init__(self):
        self.pan = None
        self.tilt = None
        self.last_error = (None, None)
        self.deadzone_hit = False
        self.last_control_time = None
        self.command_pending = False

    def reset(self):
        self.__init__()

    def update(self, box, image_size, pose, now):
        width, height = image_size
        dx = (box[0] + box[2]) * 0.5 - width * 0.5
        dy = (box[1] + box[3]) * 0.5 - height * 0.5
        self.last_error = (dx, dy)
        self.deadzone_hit = abs(dx) <= DEAD_X and abs(dy) <= DEAD_Y
        if self.pan is None:
            self.pan = float(pose["gimbal_pan"])
            self.tilt = float(pose["gimbal_tilt"])
        if self.deadzone_hit:
            return self.pan, self.tilt
        if (self.last_control_time is not None
                and now - self.last_control_time < CONTROL_INTERVAL_S):
            return self.pan, self.tilt
        focal = width / (2.0 * math.tan(math.radians(float(pose["gimbal_fov_deg"])) / 2.0))
        horizontal = math.degrees(math.atan(dx / focal))
        vertical = math.degrees(math.atan(dy / focal))
        changed = False
        if abs(dx) > DEAD_X:
            self.pan += max(-PAN_MAX_STEP_DEG,
                            min(PAN_MAX_STEP_DEG, GAIN * horizontal))
            changed = True
        if abs(dy) > DEAD_Y:
            self.tilt -= max(-TILT_MAX_STEP_DEG,
                             min(TILT_MAX_STEP_DEG, GAIN * vertical))
            changed = True
        if changed:
            self.pan = max(-180.0, min(180.0, self.pan))
            self.tilt = max(-90.0, min(20.0, self.tilt))
            self.last_control_time = now
            self.command_pending = True
        return self.pan, self.tilt




# ===== V4 来源：v4_motion.py =====
class SingleEntityMotion:
    def __init__(self, parameters=None):
        self.parameters = parameters or LocalMotionParameters()
        self.reset()

    def reset(self):
        self._prev_gray = None
        self._prev_box = None
        self._prev_all_boxes = ()
        self._prev_time = None
        self._transitions = deque(maxlen=self.parameters.window_transitions)
        self._last_raw = "UNKNOWN"
        self.decision = "UNKNOWN"
        self.evidence = {"decision": self.decision, "reason": "reset"}

    def _flow(self, gray, all_boxes):
        previous_mask = LocalMotionDetector._background_mask(
            gray.shape, self._prev_all_boxes, (self._prev_box,))
        corners = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=self.parameters.max_corners,
            qualityLevel=0.01, minDistance=7, blockSize=7, mask=previous_mask)
        empty = np.empty((0, 2), np.float32)
        if corners is None:
            return empty, empty
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, corners, None, winSize=(31, 31), maxLevel=4)
        if forward is None or ok is None:
            return empty, empty
        valid = (ok[:, 0] == 1) & np.isfinite(forward[:, 0]).all(axis=1)
        corners, forward = corners[valid], forward[valid]
        if not len(corners):
            return empty, empty
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, forward, None, winSize=(31, 31), maxLevel=4)
        if backward is None or backward_ok is None:
            return empty, empty
        old, new, back = corners[:, 0], forward[:, 0], backward[:, 0]
        good = ((backward_ok[:, 0] == 1) & np.isfinite(back).all(axis=1)
                & (np.linalg.norm(old - back, axis=1) < self.parameters.fb_error_px))
        mask = LocalMotionDetector._background_mask(gray.shape, all_boxes)
        x = np.clip(np.rint(new[:, 0]).astype(np.int32), 0, gray.shape[1] - 1)
        y = np.clip(np.rint(new[:, 1]).astype(np.int32), 0, gray.shape[0] - 1)
        good &= mask[y, x] != 0
        good &= (new[:, 0] >= 0) & (new[:, 0] < gray.shape[1])
        good &= (new[:, 1] >= 0) & (new[:, 1] < gray.shape[0])
        return old[good], new[good]

    def _transition(self, old_box, box, old_points, new_points):
        h, w = self._prev_gray.shape
        xa, ya, xb, yb = LocalMotionDetector._local_roi(old_box, w, h)
        selected = ((old_points[:, 0] >= xa) & (old_points[:, 0] < xb)
                    & (old_points[:, 1] >= ya) & (old_points[:, 1] < yb))
        p0, p1 = old_points[selected], new_points[selected]
        count = len(p0)
        if count < self.parameters.min_local_points:
            return {"decision": self.decision, "reason": "insufficient_local_features",
                    "local_feature_count": count}
        affine, inlier_mask = cv2.estimateAffine2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=1000, confidence=0.99, refineIters=10)
        if affine is None or inlier_mask is None or not np.isfinite(affine).all():
            return {"decision": self.decision, "reason": "affine_failed"}
        inliers = inlier_mask[:, 0].astype(bool)
        n = int(np.sum(inliers))
        ratio = n / count
        estimate = p0[inliers] @ affine[:, :2].T + affine[:, 2]
        rmse = float(np.sqrt(np.mean(np.sum((estimate - p1[inliers]) ** 2, axis=1)))) if n else math.inf
        evidence = {"local_feature_count": count, "affine_inlier_count": n,
                    "affine_inlier_ratio": ratio, "affine_rmse_px": rmse}
        if (n < self.parameters.min_inliers or ratio <= self.parameters.min_inlier_ratio
                or rmse >= self.parameters.max_rmse_px):
            return {"decision": self.decision, "reason": "weak_local_affine", **evidence}
        self._transitions.append((affine.copy(), _anchor(old_box), _anchor(box),
                                  float(box[3] - box[1]), rmse))
        steps = tuple(self._transitions)
        evidence["window_valid_frames"] = len(steps)
        if len(steps) < self.parameters.window_transitions:
            return {"decision": self.decision, "reason": "warmup", **evidence}
        predicted = steps[0][1].copy()
        previous = None
        for matrix, old_anchor, new_anchor, _, _ in steps:
            if previous is not None:
                predicted += old_anchor - previous
            predicted = matrix[:, :2] @ predicted + matrix[:, 2]
            previous = new_anchor
        error = float(np.linalg.norm(steps[-1][2] - predicted))
        height = float(np.median([step[3] for step in steps]))
        sigma = float(np.median([step[4] for step in steps]))
        static_limit = max(0.06 * height, 3.0, 2.0 * sigma)
        moving_limit = max(0.18 * height, 6.0, 4.0 * sigma)
        raw = ("STATIC" if error < static_limit else
               "MOVING" if error > moving_limit else "UNKNOWN")
        confirmed = raw if raw != "UNKNOWN" and raw == self._last_raw else "UNKNOWN"
        self._last_raw = raw
        if confirmed != "UNKNOWN":
            self.decision = confirmed
        evidence.update({"decision": self.decision, "raw_decision": raw,
                         "confirmed_decision": confirmed, "endpoint_error_px": error,
                         "bbox_height_px": height, "T_static": static_limit,
                         "T_moving": moving_limit, "reason": "confirmed" if confirmed != "UNKNOWN"
                         else "awaiting_confirmation"})
        return evidence

    def update(self, image_bgr, box, other_boxes, time_s):
        """仅实体可见时调用；缺帧时调用方不触碰窗口。"""
        if box is None or image_bgr is None:
            return self.evidence
        time_s = float(time_s)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        all_boxes = tuple([box, *other_boxes])
        if (self._prev_gray is None or gray.shape != self._prev_gray.shape
                or self._prev_time is None or time_s <= self._prev_time
                or time_s - self._prev_time > self.parameters.max_transition_gap_s):
            # 时间断层只更新参考帧，不删除已确认的运动窗口。
            self._prev_gray, self._prev_box = gray, box
            self._prev_all_boxes, self._prev_time = all_boxes, time_s
            self.evidence = {"decision": self.decision, "reason": "first_or_discontinuous_frame",
                             "window_valid_frames": len(self._transitions)}
            return self.evidence
        old_points, new_points = self._flow(gray, all_boxes)
        self.evidence = self._transition(self._prev_box, box, old_points, new_points)
        self._prev_gray, self._prev_box = gray, box
        self._prev_all_boxes, self._prev_time = all_boxes, time_s
        return self.evidence



def _ground_projection(center, image_size, own_pose):
    """只按本机姿态把像素射线投到高度零的粗平面。"""
    if own_pose is None:
        return None, None
    keys = ("lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg")
    try:
        own = {key: float(own_pose[key]) for key in keys}
    except (KeyError, TypeError, ValueError):
        return None, None
    if not all(math.isfinite(value) for value in own.values()) or own["alt"] <= 0:
        return None, None
    ray, _ = pixel_ray(center[0], center[1], image_size[0], image_size[1], own)
    if ray[2] >= -1e-6:
        return None, None
    scale = own["alt"] / -ray[2]
    east, north = ray[0] * scale, ray[1] * scale
    cos_lat = math.cos(math.radians(own["lat"]))
    if abs(cos_lat) < 1e-6:
        return None, None
    lat = own["lat"] + north / 111320.0
    lon = own["lon"] + east / (111320.0 * cos_lat)
    distance = math.sqrt(east * east + north * north + own["alt"] * own["alt"])
    return (lat, lon, 0.0), distance



# ===== V4 来源：v4_position.py =====
class RoughPosition:
    def __init__(self):
        self.points = deque(maxlen=5)

    def reset(self):
        self.points.clear()

    @property
    def position(self):
        if not self.points:
            return None
        return median(point[0] for point in self.points), median(point[1] for point in self.points)

    def update(self, box, image_size, pose):
        center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        point, _ = _ground_projection(center, image_size, pose)
        if point is not None:
            self.points.append((point[0], point[1]))
        return self.position




# ===== V4 来源：v4_flight.py =====
def visual_waypoint(pose, box, image_size, distance_m=180.0, origin_position=None):
    center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
    ray, _ = pixel_ray(center[0], center[1], image_size[0], image_size[1], pose)
    bearing = (math.atan2(ray[0], ray[1]) if math.hypot(ray[0], ray[1]) > 1e-6
               else math.radians(float(pose["heading_deg"]) + float(pose["gimbal_pan"])))
    origin = origin_position or (pose["lat"], pose["lon"])
    return offset_position(origin,
                           distance_m * math.sin(bearing), distance_m * math.cos(bearing))

def orbit_waypoint(target, phase_deg, role, radius_m=130.0):
    phase = math.radians(float(phase_deg))
    if role == "FOLLOWER":
        phase += math.pi
    return offset_position(target, radius_m * math.sin(phase),
                           radius_m * math.cos(phase))

def follower_ready(own, master, entry_slot):
    # 主机 130 米槽位与从机 180 米入口对置时，计划间距约 310 米。
    return (master is not None and entry_slot is not None
            and ground_distance_m(own, entry_slot) < 30.0
            and ground_distance_m(own, master) > 280.0)




# ===== V4 来源：v4_control.py =====
STATES = ("SEARCH", "VERIFY", "CALLING", "FOLLOWER_APPROACH", "COOP_TRACK")

MEMBERS = ("20001", "20002", "20003")

COOP_ORBIT_RADIUS_M = 130.0

FOLLOWER_ENTRY_RADIUS_M = 180.0

COOP_ORBIT_PERIOD_S = 60.0

COOP_SPEED_MPS = 35.0

SEARCH_CONFIRM_FRAMES = 3

VERIFY_TIMEOUT_S = 2.0

class V4Control:
    def __init__(self, uid):
        self.uid = str(uid)
        self.state = "SEARCH"
        self.entity = EntityManager(uid)
        self.motion = SingleEntityMotion()
        self.gimbal = VisualGimbal()
        self.rough = RoughPosition()
        self.coord = V4Coordinator(uid)
        self.route = CoordinatedSweepRoute(_BBOX, self.uid, MEMBERS)
        self.search_gimbal = SearchGimbalController()
        self.last_visual_box = None
        self.last_visual_size = None
        self.last_visual_pose = None
        self.last_frame_id = None
        self.events = []
        self.completed_sessions = 0
        self._last_coverage_s = -1e9
        self._last_report_s = -1e9
        self._entry_phase_deg = None
        self._orbit_phase_deg = None
        self._orbit_start_s = None
        self._search_candidate_box = None
        self._search_candidate_frames = 0
        self._verify_started_s = None
        self.last_own_position = None

    def _event(self, name, now, **details):
        self.events.append({"event": name, "time": float(now), "uid": self.uid,
                            "state": self.state, **details})

    def pop_events(self):
        events, self.events = self.events, []
        return events

    def _state(self, state, now, reason):
        if self.state != state:
            old = self.state
            self.state = state
            self._verify_started_s = None
            if old == "SEARCH":
                self._search_candidate_box = None
                self._search_candidate_frames = 0
            self._event("state_changed", now, previous=old, reason=reason)

    def _return_search(self, now, reason, terminal=None):
        if terminal:
            self.coord.queue_message(terminal, now)
        self.coord.clear_session()
        self.entity.clear()
        self.motion.reset()
        self.rough.reset()
        self.gimbal.reset()
        self.last_visual_box = None
        self._entry_phase_deg = None
        self._orbit_phase_deg = None
        self._orbit_start_s = None
        self._search_candidate_box = None
        self._search_candidate_frames = 0
        if self.state != "VERIFY":
            self.route.pause()
        self._state("SEARCH", now, reason)

    def _update_search_candidate(self, objects, image_size):
        real = [item for item in objects if item.class_name == "real_vehicle"]
        if not real:
            self._search_candidate_box = None
            self._search_candidate_frames = 0
            return None
        center = lambda box: ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        last_box = self._search_candidate_box
        if last_box is None:
            width, height = image_size
            candidate = min(real, key=lambda item: math.dist(
                center(item.bbox_xyxy), (width * 0.5, height * 0.5)))
            self._search_candidate_frames = 1
        else:
            last_center = center(last_box)
            candidate = min(real, key=lambda item: math.dist(
                center(item.bbox_xyxy), last_center))
            diagonal = math.hypot(last_box[2] - last_box[0], last_box[3] - last_box[1])
            if math.dist(center(candidate.bbox_xyxy), last_center) <= max(220.0, 1.5 * diagonal):
                self._search_candidate_frames += 1
            else:
                width, height = image_size
                candidate = min(real, key=lambda item: math.dist(
                    center(item.bbox_xyxy), (width * 0.5, height * 0.5)))
                self._search_candidate_frames = 1
        self._search_candidate_box = candidate.bbox_xyxy
        return candidate if self._search_candidate_frames >= SEARCH_CONFIRM_FRAMES else None

    def consume_visual(self, snapshot):
        """外层保证同一 frame_id 只调用一次。"""
        now = float(snapshot.source_sim_time)
        self.last_frame_id = snapshot.frame_id
        self._event("visual_frame", now, frame_id=snapshot.frame_id,
                    raw_count=len(snapshot.raw_yolo_objects),
                    effective_count=len(snapshot.effective_yolo_objects))
        if snapshot.error:
            self._event("perception_error", now, error=snapshot.error)
            if self.state == "SEARCH":
                self._search_candidate_box = None
                self._search_candidate_frames = 0
            return
        if self.state == "FOLLOWER_APPROACH" or (
                self.state == "COOP_TRACK" and self.coord.master_uid != self.uid):
            return
        if self.state == "SEARCH":
            candidate = self._update_search_candidate(snapshot.effective_yolo_objects,
                                                      snapshot.image_size)
            if candidate is None:
                return
            entity, event = self.entity.update((candidate,), snapshot.image_size, now)
            self.motion.reset()
            self.rough.reset()
            self._state("VERIFY", now, "real_vehicle_three_frames")
        else:
            entity, event = self.entity.update(snapshot.effective_yolo_objects,
                                               snapshot.image_size, now)
        if event:
            self._event(event, now, frame_id=snapshot.frame_id,
                        entity_id=entity.entity_id, entity_visible=entity.visible,
                        entity_bbox=entity.bbox_xyxy, entity_missing_s=entity.missing_s,
                        entity_observed_frames=entity.observed_frames)
        if event == "entity_lost":
            self._return_search(now, "entity_lost",
                                "CANCEL" if self.state in ("CALLING", "COOP_TRACK") else None)
            return
        if entity is None or not entity.visible:
            return
        self.last_visual_box = entity.bbox_xyxy
        self.last_visual_size = snapshot.image_size
        self.last_visual_pose = dict(snapshot.source_pose)
        self.gimbal.update(entity.bbox_xyxy, snapshot.image_size, snapshot.source_pose, now)
        other = [item.bbox_xyxy for item in snapshot.effective_yolo_objects
                 if item.bbox_xyxy != entity.bbox_xyxy]
        previous_motion = self.motion.decision
        motion = self.motion.update(snapshot.image_bgr, entity.bbox_xyxy, other, now)
        if self.motion.decision != previous_motion:
            self._event("motion_changed", now, entity_id=entity.entity_id,
                        motion_decision=self.motion.decision, motion_evidence=motion)
        if self.state == "VERIFY":
            if self.motion.decision == "STATIC":
                self._return_search(now, "motion_static")
                return
            if entity.observed_frames >= 3 and self.motion.decision == "MOVING":
                self._state("CALLING", now, "motion_moving")
                self.coord.set_master(entity.entity_id)
        if self.state in ("CALLING", "COOP_TRACK"):
            self.rough.update(entity.bbox_xyxy, snapshot.image_size, snapshot.source_pose)
            if self.state == "COOP_TRACK" and self.motion.decision == "STATIC":
                self.completed_sessions += 1
                self._return_search(now, "motion_static_completed", "DONE")

    def _own_position(self, obs):
        return float(obs.self.lat), float(obs.self.lon)

    def _heartbeat_state(self):
        if self.state == "COOP_TRACK":
            return "COOP_TRACK_M" if self.coord.master_uid == self.uid else "COOP_TRACK_F"
        return self.state

    def _orbit_waypoint(self, target, now, role):
        if self._orbit_phase_deg is None or self._orbit_start_s is None:
            return None
        phase = (self._orbit_phase_deg
                 + 360.0 * (now - self._orbit_start_s) / COOP_ORBIT_PERIOD_S)
        return orbit_waypoint(target, phase, role, COOP_ORBIT_RADIUS_M)

    def step(self, obs, now):
        """消息跨 tick 生效；每拍只从本机观测生成本机 commands。"""
        now = float(now)
        own = self._own_position(obs)
        self.last_own_position = own
        pose = {key: float(getattr(obs.self, key)) for key in (
            "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt",
            "gimbal_fov_deg")}
        incoming = self.coord.ingest(obs.comm_inbox, now, self.state)
        for kind in incoming:
            self._event("comm_in", now, message_kind=kind, session=self.coord.session)
            if kind == "INVITE" and self.state == "SEARCH":
                self._state("FOLLOWER_APPROACH", now, "invite_received")
                self.coord.queue_message("ACCEPT", now)
            elif kind == "READY" and self.state == "CALLING":
                target = self.rough.position
                if target is not None:
                    self._orbit_phase_deg = bearing_deg(target, own)
                    self._orbit_start_s = now
                    self._state("COOP_TRACK", now, "follower_ready")
                    self.coord.queue_message("START", now,
                                             phase_deg=self._orbit_phase_deg,
                                             start_s=self._orbit_start_s)
            elif kind == "START" and self.state == "FOLLOWER_APPROACH":
                self._orbit_phase_deg = self.coord.orbit_phase_deg
                self._orbit_start_s = self.coord.orbit_start_s
                self._state("COOP_TRACK", now, "start_received")
            elif kind in ("DONE", "CANCEL") and self.coord.master_uid != self.uid:
                self._return_search(now, kind.lower())
        if (self.state in ("FOLLOWER_APPROACH", "COOP_TRACK")
                and self.coord.master_uid != self.uid
                and now - self.coord.last_master_message_s > 5.0):
            self._return_search(now, "master_communication_timeout")
        if self.state == "VERIFY":
            if self._verify_started_s is None:
                self._verify_started_s = now
            elif now - self._verify_started_s >= VERIFY_TIMEOUT_S:
                self._return_search(now, "verify_timeout")
        self.coord.queue_message("H", now, position=own,
                                 state=self._heartbeat_state(), period_s=1.0)
        commands = []
        if self.state in ("SEARCH", "VERIFY"):
            self.route.observe_search_position(own)
            if now - self._last_coverage_s >= 0.5:
                self.route.coverage.observe(
                    now, own, pose["heading_deg"], pose["gimbal_pan"],
                    pose["gimbal_tilt"], pose["gimbal_fov_deg"])
                self._last_coverage_s = now
            target = self.route.target(own, now, self.coord.peers)
            for name, details in self.route.drain_events():
                self._event(name, now, **details)
            heading = pose["heading_deg"]
            if target is not None:
                heading = bearing_deg(own, target)
                delta = abs((heading - pose["heading_deg"] + 180.0) % 360.0 - 180.0)
                speed = 15.0 if delta > 45.0 else 22.0
                commands.append(fly_to(*target, alt=500.0, speed=speed, loiter_radius=0.0))
            if self.state == "SEARCH":
                pan, tilt = self.search_gimbal.scan(
                    now, heading, pose["heading_deg"], pose["gimbal_pan"], pose["gimbal_tilt"])
                commands.append(point_gimbal(pan, tilt))
            elif self.gimbal.command_pending:
                commands.append(point_gimbal(self.gimbal.pan, self.gimbal.tilt))
                self.gimbal.command_pending = False
            commands.append(set_gimbal_fov(48.0))
        else:
            self.route.pause()
            if self.state == "CALLING" or (
                    self.state == "COOP_TRACK" and self.coord.master_uid == self.uid):
                if self.last_visual_box is not None and self.last_visual_size is not None:
                    if self.state == "COOP_TRACK" and self.rough.position is not None:
                        target = self._orbit_waypoint(self.rough.position, now, "MASTER")
                    elif self.state == "CALLING" and self.rough.position is not None:
                        if self._entry_phase_deg is None:
                            self._entry_phase_deg = bearing_deg(self.rough.position, own)
                        target = orbit_waypoint(self.rough.position, self._entry_phase_deg,
                                                "MASTER", COOP_ORBIT_RADIUS_M)
                    else:
                        target = visual_waypoint(self.last_visual_pose, self.last_visual_box,
                                                 self.last_visual_size, origin_position=own)
                    if target is not None:
                        speed = COOP_SPEED_MPS if self.state in ("CALLING", "COOP_TRACK") else 22.0
                        commands.append(fly_to(*target, alt=500.0, speed=speed,
                                               loiter_radius=0.0))
                if self.gimbal.command_pending:
                    commands.append(point_gimbal(self.gimbal.pan, self.gimbal.tilt))
                    self.gimbal.command_pending = False
            else:
                target = self.coord.target
                if target is not None:
                    if self.state == "COOP_TRACK":
                        destination = self._orbit_waypoint(target, now, "FOLLOWER")
                    else:
                        master = self.coord.master_position
                        phase = bearing_deg(target, master) if master is not None else None
                        destination = (orbit_waypoint(target, phase, "FOLLOWER",
                                                      FOLLOWER_ENTRY_RADIUS_M)
                                       if phase is not None else None)
                    if destination is not None:
                        commands.append(fly_to(*destination, alt=500.0, speed=COOP_SPEED_MPS,
                                               loiter_radius=0.0))
                    aim = solve_ground_aim(own, pose["alt"], pose["heading_deg"], target)
                    commands.append(point_gimbal(aim.pan_deg, aim.tilt_deg))
                    if self.state == "FOLLOWER_APPROACH" and follower_ready(
                            own, self.coord.master_position, destination):
                        self.coord.queue_message("READY", now, period_s=1.0)
            commands.append(set_gimbal_fov(48.0))
        if self.state == "CALLING":
            peer = self.coord.peers.get(self.coord.partner_uid)
            if (not self.coord.accepted and self.coord.partner_uid is not None
                    and (peer is None or now - peer[1] > 5.0
                         or peer[2] not in ("SEARCH", "FOLLOWER_APPROACH"))):
                self.coord.partner_uid = None
            if self.coord.partner_uid is None:
                self.coord.partner_uid = self.coord.select_partner(
                    own, now, allowed_uids=self.route.preferred_partner_uids())
            if (not self.coord.accepted and self.coord.partner_uid is not None
                    and self.rough.position is not None):
                self.coord.queue_message("INVITE", now, target=self.rough.position,
                                         period_s=1.0)
            if self.coord.accepted and self.rough.position is not None:
                self.coord.queue_message("TARGET", now, target=self.rough.position,
                                         position=own, period_s=0.5)
        elif self.state == "COOP_TRACK" and self.coord.master_uid == self.uid:
            target = self.rough.position
            if target is not None:
                self.coord.queue_message("TARGET", now, target=target,
                                         position=own, period_s=0.5)
                if now - self._last_report_s >= 1.0:
                    commands.append(report_target(*target))
                    self._last_report_s = now
            self.coord.queue_message("START", now, phase_deg=self._orbit_phase_deg,
                                     start_s=self._orbit_start_s, period_s=1.0)
        elif self.state == "FOLLOWER_APPROACH":
            self.coord.queue_message("ACCEPT", now, period_s=1.0)
        command, kind = self.coord.emit(now)
        if command is not None:
            commands.append(command)
            self._event("comm_out", now, message_kind=kind, session=self.coord.session)
        return commands



@dataclass(frozen=True)
class V4Object:
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    real_probability: float
    decoy_probability: float
    class_name: str


@dataclass(frozen=True)
class V4Snapshot:
    uid: str
    frame_id: str
    source_sim_time: float
    image_size: tuple[int, int]
    source_pose: Mapping[str, float]
    raw_yolo_objects: tuple[V4Object, ...]
    effective_yolo_objects: tuple[V4Object, ...]
    image_bgr: np.ndarray | None = None
    error: str | None = None


def _letterbox(image, shape):
    height, width = image.shape[:2]
    out_h, out_w = shape
    gain = min(out_h / height, out_w / width)
    new_w, new_h = round(width * gain), round(height * gain)
    left, top = (out_w - new_w) // 2, (out_h - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, top, out_h - new_h - top,
                                left, out_w - new_w - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
    return rgb, (gain, left, top)


def _decode(raw, original_shape, geometry):
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 6:
        raise ValueError("模型输出不是双类别原始预测")
    anchors = raw[0].transpose(0, 1).float()
    scores = anchors[:, 4:6].amax(1)
    keep = (scores >= 0.05) & torch.isfinite(anchors).all(1)
    anchors, scores = anchors[keep], scores[keep]
    if not len(anchors):
        return np.empty((0, 7), dtype=np.float32)
    boxes = torch.cat((anchors[:, :2] - anchors[:, 2:4] / 2,
                       anchors[:, :2] + anchors[:, 2:4] / 2), dim=1)
    selected = nms(boxes, scores, 0.5)[:100]
    boxes = boxes[selected]
    probs = anchors[selected, 4:6]
    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-9)
    gain, left, top = geometry
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_shape[0])
    output = torch.cat((boxes, scores[selected, None], probs), dim=1)
    output = output[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    return output.cpu().numpy()


class _Detector:
    def __init__(self):
        from ultralytics import YOLO
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(Path(__file__).resolve().with_name("yolo.pt"))).model
        if self.model.names != {0: "real_vehicle", 1: "model_prop"}:
            raise ValueError("模型类别与 V4 不一致")
        self.model.end2end = False
        self.model = self.model.to(self.device).float().eval().fuse(verbose=False)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model.to(dtype=self.dtype)

    @torch.inference_mode()
    def predict(self, image):
        rgb, geometry = _letterbox(image, (1152, 1536))
        tensor = torch.from_numpy(rgb).unsqueeze(0).to(self.device, dtype=self.dtype) / 255
        return _decode(self.model(tensor)[0], image.shape[:2], geometry)


class _VisionWorker:
    """三架 Agent 共用模型；每架只保留尚未处理的最新照片。"""
    def __init__(self):
        self.condition = Condition()
        self.pending = {}
        self.completed = {}
        self.last_submitted = {}
        self.failure = None
        Thread(target=self._run, daemon=True, name="v9aget-vision").start()

    def submit(self, key, uid, photo, now, pose):
        if not isinstance(photo, bytes) or not photo:
            return
        frame_id = hashlib.sha256(photo).hexdigest()
        with self.condition:
            if self.last_submitted.get(key) == frame_id:
                return
            self.last_submitted[key] = frame_id
            self.pending[key] = (uid, frame_id, photo, now, pose)
            self.condition.notify()

    def latest(self, key, now):
        with self.condition:
            snapshot = self.completed.get(key)
        if snapshot is None or not 0 <= now - snapshot.source_sim_time <= 1.5:
            return None
        return snapshot

    def _run(self):
        try:
            detector = _Detector()
        except Exception as exc:
            with self.condition:
                self.failure = repr(exc)
            return
        while True:
            with self.condition:
                while not self.pending:
                    self.condition.wait()
                key = next(iter(self.pending))
                uid, frame_id, photo, now, pose = self.pending.pop(key)
            try:
                image = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("无法解码照片")
                rows = detector.predict(image)
                height, width = image.shape[:2]
                objects = []
                for row in rows:
                    box = tuple(float(value) for value in row[:4])
                    real, decoy = float(row[5]), float(row[6])
                    if not (0 <= box[0] < box[2] <= width and
                            0 <= box[1] < box[3] <= height):
                        continue
                    class_name = ("real_vehicle" if real >= decoy else "model_prop") if max(
                        real, decoy) >= 0.6 else "uncertain"
                    objects.append(V4Object(box, float(row[4]), real, decoy, class_name))
                found = tuple(objects)
                snapshot = V4Snapshot(uid, frame_id, now, (width, height), pose,
                                      found, found, image)
            except Exception as exc:
                snapshot = V4Snapshot(uid, frame_id, now, (0, 0), pose,
                                      (), (), error=repr(exc))
            with self.condition:
                self.completed[key] = snapshot


_worker_lock = Lock()
_worker = None


def _vision_worker():
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _VisionWorker()
        return _worker


class V9aget(CoopAgent):
    """v9aget：V4 三机协同识别正式提交版。"""
    def __init__(self, my_uid):
        super().__init__(my_uid)
        self.reset()

    def reset(self):
        self.control = V4Control(self.my_uid)
        self._vision_key = object()
        self._last_snapshot = None
        self._consumed_frame_id = None
        self._t = 0.0

    def sensor(self, obs, dt):
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        now = float(score.sim_time) if score is not None else self._t + max(0.0, float(dt))
        photo = obs.self.photo
        if photo:
            worker = _vision_worker()
            pose = {key: float(getattr(obs.self, key)) for key in (
                "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt",
                "gimbal_fov_deg")}
            worker.submit(self._vision_key, str(self.my_uid), photo, now, pose)
        else:
            worker = _worker
        if worker is None:
            return []
        if worker.failure is not None:
            raise RuntimeError(f"视觉模型初始化失败：{worker.failure}")
        snapshot = worker.latest(self._vision_key, now)
        if snapshot is None:
            return []
        self._last_snapshot = snapshot
        if snapshot.error:
            return []
        real = [item for item in snapshot.effective_yolo_objects
                if item.class_name == "real_vehicle"]
        if not real:
            return []
        width, height = snapshot.image_size
        item = min(real, key=lambda obj: ((obj.bbox_xyxy[0] + obj.bbox_xyxy[2] - width) ** 2
                                         + (obj.bbox_xyxy[1] + obj.bbox_xyxy[3] - height) ** 2))
        box = item.bbox_xyxy
        point, _ = _ground_projection(((box[0] + box[2]) * 0.5,
                                       (box[1] + box[3]) * 0.5),
                                      snapshot.image_size, snapshot.source_pose)
        if point is None:
            return []
        return [Detection(detected=True,
                          confidence=item.detector_confidence * item.real_probability,
                          target_lat=point[0], target_lon=point[1],
                          target_type="ground_vehicle")]

    def decide(self, obs, dt):
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        now = float(score.sim_time) if score is not None else self._t + max(0.0, float(dt))
        self._t = now
        snapshot = self._last_snapshot
        if snapshot is not None and snapshot.frame_id != self._consumed_frame_id:
            self._consumed_frame_id = snapshot.frame_id
            self.control.consume_visual(snapshot)
        commands = self.control.step(obs, now)
        self.control.pop_events()
        return commands
