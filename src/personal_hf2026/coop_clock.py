# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
"""按双机实际观测时间配对，估计同一目标的协同盯防时长。"""

from collections import deque

from competition.baselines.coop_distributed import _haversine_m


class CoopClock:
    # 控制约 10Hz、报文时间保留一位小数，配对容差覆盖采样和量化误差。
    PAIR_TOLERANCE_S = 0.15
    # 仅针对零噪声控制测试：量化和短时运动允许少量误差，不能混算邻近车辆。
    PAIR_DISTANCE_M = 10.0
    MAX_REPORT_AGE_S = 1.0
    GAP_S = 2.0

    def __init__(self):
        self.history = deque()
        self.last_seq = -1
        self.last_pair = None
        self.seconds = 0.0
        self.peak_seconds = 0.0
        self.resets = 0
        self.updated = False

    def observe(self, now, position, inbox):
        self.updated = False
        while self.history and now - self.history[0][0] > 3.0:
            self.history.popleft()
        if position is not None and (not self.history or now > self.history[-1][0]):
            self.history.append((now, position))
        for message in inbox:
            if message.sender_uid != "20002":
                continue
            fields = message.payload.split(",")
            if len(fields) != 6 or fields[0] != "1":
                continue
            try:
                seq, seen = int(fields[1]), float(fields[2])
                peer_pos = (float(fields[4]), float(fields[5]))
            except ValueError:
                continue
            if seq <= self.last_seq:
                continue
            self.last_seq = seq
            # 报文四舍五入允许最多 0.05 秒超前，旧检测不能当作新证据。
            if not -0.05 <= now - seen <= self.MAX_REPORT_AGE_S:
                continue
            matches = [(abs(t - seen), t) for t, pos in self.history
                       if abs(t - seen) <= self.PAIR_TOLERANCE_S
                       and _haversine_m(*pos, *peer_pos) < self.PAIR_DISTANCE_M]
            if not matches:
                continue
            paired_at = min(seen, min(matches)[1], now)
            if self.last_pair is not None and paired_at <= self.last_pair:
                continue
            if self.last_pair is not None and paired_at - self.last_pair <= self.GAP_S:
                self.seconds += paired_at - self.last_pair
            else:
                if self.seconds > 0:
                    self.resets += 1
                self.seconds = 0.0
            self.last_pair = paired_at
            self.peak_seconds = max(self.peak_seconds, self.seconds)
            self.updated = True
        # 等待迟到报文时暂停，不补空等时间；新证据仍须通过两秒间隔检查。
        if self.last_pair is not None and now - self.last_pair > self.GAP_S + self.MAX_REPORT_AGE_S:
            if self.seconds > 0:
                self.resets += 1
            self.seconds = 0.0
            self.last_pair = None
