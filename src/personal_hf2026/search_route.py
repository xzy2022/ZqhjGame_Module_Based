# 修改时间：2026-09-24。
# 修改目的：避免搜索中断后命令无人机返回中断前的位置。
# 修改内容：暂停时仅停止航点推进，并移除旧位置恢复航点。
# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：用可执行且可恢复的条带航线替代按时间旋转的搜索航点。
# 修改内容：管理分区、固定航点、到点推进、中断恢复及实际位置网格访问。
"""独立的矩形分区搜索航线，不依赖目标真值或协同状态机。"""

import math


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
