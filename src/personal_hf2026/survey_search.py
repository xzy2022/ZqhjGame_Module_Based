# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：将每架无人机的普查航线由三条缩减为两条。
# 修改内容：将最大条带间距调整为七百米，使当前三分区实际生成约六百六十一米间距。
# 修改时间：2026-09-13
# 修改目的：让有限时间搜索优先遍历长轴，再按实际观测留下的空白补扫。
# 修改内容：新增保守视野覆盖网格、宽间距普查航线和连续漏扫航段选择。
"""普查与补扫管理，不读取车辆真值、地形或其他飞机的内部变量。"""

import math

from .search_route import StripSearchRoute


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
        self.supplement_legs+=1

    def target(self, position, heading=0.0):
        if not self.waypoints:
            self.initial_heading=heading
            self._build(position)
        self._active=True
        self._last_route_position=position
        if self._resume_position is not None:
            if self._distance(position,self._resume_position)>self.arrival_radius_m:
                return self._resume_position
            self._resume_position=None
        if self._distance(position,self.waypoints[self.index])<=self.arrival_radius_m:
            self.completed_waypoints+=1
            self.index+=1
            if self.index==len(self.waypoints):
                if self.phase=="SURVEY":
                    self.survey_completed=True
                    self.completed_passes+=1
                self._next_supplement(position)
        return self.waypoints[self.index]

    def clamp_position(self, position):
        return tuple(max(self.bounds[0][i],min(self.bounds[1][i],position[i])) for i in (0,1))

    @property
    def summary(self):
        return dict(super().summary, phase=self.phase, axis="NS" if self.north_south else "EW",
                    actual_spacing_m=self.actual_spacing_m, survey_completed=self.survey_completed,
                    supplement_legs=self.supplement_legs, coverage=self.coverage.summary)
