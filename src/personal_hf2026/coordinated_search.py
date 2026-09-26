# 修改时间：2026-09-26。
# 修改目的：让三架无人机按初始空间顺序协同扫描整张地图。
# 修改内容：新增短轴三分区、对齐、Z 字搜索、中区接管与从当前位置恢复的轻量规划器。
"""仅根据本机位置与同伴心跳规划三机搜索航点。"""

from __future__ import annotations

import math

from .survey_search import SearchCoverageGrid


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
