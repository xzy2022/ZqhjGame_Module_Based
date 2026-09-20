# 修改时间：2026-09-20（正式感知与双机融合接线）。
# 修改目的：让 V3 标准 Agent 直接消费共享像素 worker，并在协同阶段按通信预算完成估高上报。
# 修改内容：接入零高程控制投影、F3 心跳槽替换、双机定位及仅估高成功后的 1Hz 上报。
# 修改时间：2026-09-20。
# 修改目的：提供可由正式 Runner 直接实例化并延迟注入真实感知服务的 V3 Agent 骨架。
# 修改内容：实现标准 sensor/decide 生命周期适配，不绑定尚未集成的具体视觉和地理融合模块。
"""正式 V3 Agent 组合入口；具体感知服务由集成层注入。"""

from collections import Counter

from competition.sdk.core.commands import broadcast, report_target

from .v3_control import PersonalV3ControlAgent
from .v3_fusion import V3Fusion, measurement_from_snapshot
from .v3_perception import snapshot_to_sensor_detections


class PersonalV3Agent(PersonalV3ControlAgent):
    """保持 ``Agent(my_uid)`` 构造契约的 V3 组合骨架。

    provider 可以是可调用对象，或实现 ``observe(obs, dt)`` / ``infer(obs, dt)``。
    返回值可为感知快照对象或字典；若没有 provider，则显式返回空列表，禁止
    SDK 默认理想感知回退进入正式路径。
    """

    perception_provider = None

    def set_perception_provider(self, provider):
        self._perception_provider = provider
        return self

    def configure(self, config):
        super().configure(config)
        self._perception_provider = self.perception_provider
        if isinstance(config, dict):
            provider = config.get("perception_provider")
            factory = config.get("perception_factory")
            if provider is not None:
                self._perception_provider = provider
            elif callable(factory):
                self._perception_provider = factory(self.my_uid)

    def reset(self):
        super().reset()
        self._fusion = V3Fusion(self.my_uid)
        self._local_measurement = None
        self._last_snapshot = None
        self._last_snapshot_key = None
        self._last_report_by_session = {}
        self._last_fusion_report_t = -1e9
        self._v3_stats = Counter()
        self._last_fusion_candidate = None
        self._pending_comm_payloads = []
        self._last_comm_send = -1e9
        provider = getattr(self, "_perception_provider", None)
        reset = getattr(provider, "reset", None)
        if callable(reset):
            reset()

    @staticmethod
    def _ground_position(observation):
        point = getattr(observation, "ground_point_h0", None)
        if point is None and isinstance(observation, dict):
            point = observation.get("ground_point_h0")
        if point is None or len(point) < 2:
            return None
        return float(point[0]), float(point[1])

    @staticmethod
    def _snapshot_field(snapshot, name, default=None):
        if isinstance(snapshot, dict):
            return snapshot.get(name, default)
        return getattr(snapshot, name, default)

    def sensor(self, obs, dt):
        provider = getattr(self, "_perception_provider", None)
        if provider is None:
            self._v3_stats["missing_provider_ticks"] += 1
            return []
        if callable(provider):
            snapshot = provider(obs, dt)
        elif callable(getattr(provider, "observe", None)):
            snapshot = provider.observe(obs, dt)
        else:
            snapshot = provider.infer(obs, dt)
        if snapshot is None:
            self._v3_stats["waiting_for_pixel_result_ticks"] += 1
            self._local_measurement = None
            return []
        snapshot_key = (
            self._snapshot_field(snapshot, "frame_id"),
            self._snapshot_field(snapshot, "source_sim_time"),
        )
        detection = self._snapshot_field(snapshot, "detection")
        track_predict = self._snapshot_field(snapshot, "track_predict")
        competitor = self._snapshot_field(snapshot, "closest_others")
        target_position = self._ground_position(detection)
        track_predict_position = self._ground_position(track_predict)
        competitor_position = self._ground_position(competitor)
        if snapshot_key != self._last_snapshot_key:
            class_name = (
                detection.get("class_name", "") if isinstance(detection, dict)
                else getattr(detection, "class_name", "")
            )
            self.submit_perception(
                snapshot,
                target_position=target_position,
                track_predict_position=track_predict_position,
                competitor_position=competitor_position,
                primary_is_target=(
                    detection is not None and class_name == "real_vehicle"
                ),
            )
            self._last_snapshot_key = snapshot_key
            try:
                source_pose = self._snapshot_field(snapshot, "source_pose")
                if source_pose is None:
                    raise ValueError("异步像素快照缺少提交时相机位姿")
                self._local_measurement = measurement_from_snapshot(
                    snapshot, source_pose
                )
            except (TypeError, ValueError):
                self._local_measurement = None
                self._v3_stats["measurement_failures"] += 1
            if self._local_measurement is not None:
                self._v3_stats["local_measurements"] += 1
        self._last_snapshot = snapshot
        return snapshot_to_sensor_detections(snapshot)

    @staticmethod
    def _broadcast_payload(command):
        if getattr(command, "verb", None) != "comm.broadcast":
            return None
        return command.params.get("payload")

    @staticmethod
    def _payload_family(payload):
        if str(payload).startswith("F3"):
            return "F3"
        return str(payload).split(",", 1)[0]

    def _schedule_broadcasts(self, commands, now):
        """统一仲裁 V1 与 F3，确保实际发出的广播最多约 3.85Hz。"""
        non_comm = []
        for command in commands:
            payload = self._broadcast_payload(command)
            if payload is None:
                non_comm.append(command)
                continue
            family = self._payload_family(payload)
            # 周期包仅保留最新值；状态切换/清除包按完整载荷去重并排队。
            if family in {"H", "T", "O", "F3"}:
                self._pending_comm_payloads = [
                    item for item in self._pending_comm_payloads
                    if self._payload_family(item) != family
                ]
            if payload not in self._pending_comm_payloads:
                self._pending_comm_payloads.append(payload)
        if now - self._last_comm_send < 0.26 or not self._pending_comm_payloads:
            return non_comm
        critical = {"G": 0, "A": 0, "C": 0, "R": 0, "K": 0}
        operational = {"T": 1, "O": 1, "F3": 2, "H": 3}
        index = min(
            range(len(self._pending_comm_payloads)),
            key=lambda item: (
                critical.get(
                    self._payload_family(self._pending_comm_payloads[item]),
                    operational.get(
                        self._payload_family(self._pending_comm_payloads[item]), 0
                    ),
                ),
                item,
            ),
        )
        payload = self._pending_comm_payloads.pop(index)
        non_comm.append(broadcast(payload))
        self._last_comm_send = now
        self._v3_stats["broadcasts_emitted"] += 1
        return non_comm

    def decide(self, obs, dt):
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        now = float(score.sim_time) if score is not None else self._t + max(0.0, dt)
        # 先保存所有合法 F3 收件；最终会按 super().decide 更新后的当前会话筛选。
        accepted = self._fusion.ingest(obs.comm_inbox, now, session=None)
        self._v3_stats["fusion_messages_accepted"] += len(accepted)
        commands = super().decide(obs, dt)

        # V1 的零高度报告只能用于理想感知实验；正式 V3 只允许双机估高后的报告。
        legacy_reports = sum(
            getattr(command, "verb", None) == "agent.report"
            for command in commands
        )
        if legacy_reports:
            self._v3_stats["legacy_reports_suppressed"] += legacy_reports
        commands = [
            command for command in commands
            if getattr(command, "verb", None) != "agent.report"
        ]

        coordinator = self._coordinator
        active = (
            coordinator.phase == coordinator.ACTIVE
            and coordinator.role in (coordinator.MASTER, coordinator.FOLLOWER)
            and coordinator.partner_uid is not None
            and coordinator.current_session is not None
        )
        heartbeat_index = next(
            (
                index for index, command in enumerate(commands)
                if str(self._broadcast_payload(command) or "").startswith("H,")
            ),
            None,
        )
        if active and heartbeat_index is not None and self._local_measurement is not None:
            payload = self._fusion.prepare_broadcast(
                now,
                coordinator.partner_uid,
                coordinator.current_session,
                self._local_measurement,
                shared_slot_granted=True,
            )
            if payload is not None:
                commands[heartbeat_index] = broadcast(payload)
                self._v3_stats["fusion_payloads_sent"] += 1

        if active and self._local_measurement is not None:
            candidate = self._fusion.report_candidate(
                self._local_measurement,
                coordinator.current_session,
                role=coordinator.role,
                peer_uid=coordinator.partner_uid,
            )
            if candidate is not None:
                self._last_fusion_candidate = candidate
                self._v3_stats["fusion_estimates"] += 1
                session_key = str(candidate["session"])
                last = self._last_report_by_session.get(session_key, -1e9)
                age = now - float(candidate["source_sim_time"])
                if (age <= 0.5 and now - last >= 1.0
                        and now - self._last_fusion_report_t >= 1.0):
                    commands.append(report_target(candidate["lat"], candidate["lon"]))
                    self._last_report_by_session[session_key] = now
                    self._last_fusion_report_t = now
                    self._v3_stats["fusion_reports_sent"] += 1
                elif age > 0.5:
                    self._v3_stats["stale_fusion_reports_suppressed"] += 1
        return self._schedule_broadcasts(commands, now)

    @property
    def completion_summary(self):
        summary = dict(super().completion_summary)
        summary.update({
            "v3_stats": dict(self._v3_stats),
            "v3_last_fusion_candidate": self._last_fusion_candidate,
        })
        return summary


__all__ = ["PersonalV3Agent"]
