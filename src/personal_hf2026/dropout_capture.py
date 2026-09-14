# 修改时间：2026-09-14
# 修改目的：让个人实验脱离官方仓库的后续修改并支持独立运行。
# 修改内容：统一模块、SDK、运行资源及输出路径并保留实验行为。
# 修改时间：2026-09-13
# 修改目的：让 Redis 失联时仍清理本轮专属渲染进程。
# 修改内容：关闭通知发送失败后继续等待或清理自有进程，不跳过资源释放。
# 修改时间：2026-09-12
# 修改目的：保证事件引用的最近帧实际落盘，并明确其时间偏差和图像内对象。
# 修改内容：补存事件最近帧及 UE 原始框元数据，移除未使用的观测缓存。
# 修改时间：2026-09-12
# 修改目的：为协同主锁定中断保留可对时的三机画面，排查遮挡与计算延迟。
# 修改内容：复用 UE 服务协议加载想定，并在后台保存中断前后相机帧及时间戳。
"""仅供本地实验的渲染与取证旁路，不向智能体提供额外信息。"""

from collections import deque
import json
from pathlib import Path

from .paths import PROJECT_ROOT, SIM_ROOT, RUNTIME_ROOT, OUTPUT_ROOT, SCENARIO_ROOT
import queue
import subprocess
import threading
import time

import redis


class DropoutCapture:
    """后台读取三机最新帧；首次出图、协同建立和丢失事件触发保存。"""

    COOP_STATES = ("COOP_INIT", "COOP_ACTIVE")

    def __init__(self, output, uids, host="127.0.0.1", port=6379):
        self.output = Path(output) / "capture"
        self.output.mkdir()
        self.uids = list(uids)
        self.redis = redis.Redis(host=host, port=port, socket_timeout=2)
        self.pending = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = None
        self.previous = {}
        self.gaps = {}
        self.sequence = 0
        self.frames = {uid: deque(maxlen=12) for uid in uids}
        self.latest = {}
        self.last_sample = {}
        self.saved = set()
        self.until = 0.0
        self.events_file = (self.output / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        self.frames_file = (self.output / "frames.jsonl").open("w", encoding="utf-8", buffering=1)
        self.stats = {"frames_saved": {}, "events": {}, "poll_errors": 0}

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="DropoutCapture", daemon=True)
        self.thread.start()

    def observe(self, row):
        """主线程只判断状态边沿并入队，不读取图像或写盘。"""
        uid = row["uid"]
        prev = self.previous.get(uid)
        summary = row["summary"]
        in_coop = row["state"] in self.COOP_STATES
        same_session = prev is not None and prev["session"] == row["session"]
        valid = bool(summary["coop_lock_valid"])
        event = None
        if uid in self.gaps:
            if not in_coop or not same_session:
                event = "lock_gap_session_ended"
            elif valid:
                event = "lock_recovered"
        elif in_coop and prev is not None and prev["state"] in self.COOP_STATES:
            # 主检测消失或跳到别的目标都取证，不能只检查 detected=False。
            if same_session and prev["summary"]["coop_lock_valid"] and not valid:
                event = "lock_gap_started"
        if event == "lock_gap_started":
            self.sequence += 1
            self.gaps[uid] = {"gap_id": self.sequence, "started_t": row["t"]}
        gap = self.gaps.get(uid)
        if event is None and in_coop and (prev is None or not same_session):
            event = "session_started"
        if event is None and prev is not None and not prev["self"]["detection"]["detected"]:
            if row["self"]["detection"]["detected"] and in_coop:
                event = "primary_detected"
        if event:
            # 保留前后两份观测，角色回退后的 NONE 不会掩盖原来的 MASTER/FOLLOWER。
            record = {"event": event, "uid": uid, "t": row["t"],
                      "sim_time": row["sim_time"], "observed_unix_s": row["observed_unix_s"],
                      "gap": dict(gap) if gap else None, "before": prev, "after": row}
            if gap:
                record["gap_duration_s"] = row["t"] - gap["started_t"]
            self.pending.put(("event", record))
        if gap or event:
            self.pending.put(("keep", row))
        if event in ("lock_recovered", "lock_gap_session_ended"):
            del self.gaps[uid]
        self.previous[uid] = row

    @staticmethod
    def _describe(frame):
        meta = {key: value for key, value in frame.items() if key != "image"}
        suffix = ".png" if frame["image"].startswith(b"\x89PNG") else ".jpg"
        meta["path"] = f"frames/{frame['uid']}/{frame['frame_no']}{suffix}"
        return meta

    def _save(self, frame):
        identity = (frame["uid"], frame["frame_no"], frame["source_sim_time"])
        if identity in self.saved:
            return
        meta = self._describe(frame)
        path = self.output / meta["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(frame["image"])
        self.frames_file.write(json.dumps(meta) + "\n")
        self.saved.add(identity)
        counts = self.stats["frames_saved"]
        counts[frame["uid"]] = counts.get(frame["uid"], 0) + 1

    def _drain(self):
        while True:
            try:
                kind, record = self.pending.get_nowait()
            except queue.Empty:
                return
            self.until = time.monotonic() + 2.0
            if kind == "keep":
                continue
            record["latest_frames"] = {
                uid: dict(self._describe(frame),
                          observation_minus_frame_s=record["sim_time"] - frame["source_sim_time"])
                for uid, frame in self.latest.items()}
            record["missing_camera_uids"] = [uid for uid in self.uids if uid not in self.latest]
            self.events_file.write(json.dumps(record) + "\n")
            counts = self.stats["events"]
            counts[record["event"]] = counts.get(record["event"], 0) + 1
            # 最近帧可能尚未达到 2 Hz 采样间隔，事件引用的帧也必须保存。
            for frame in self.latest.values():
                self._save(frame)
            for frames in self.frames.values():
                for frame in frames:
                    self._save(frame)

    def _poll(self):
        for uid in self.uids:
            keys = list(self.redis.scan_iter(match=f"sync_camera:{uid}:frame:*", count=100))
            if not keys:
                continue
            key = max(keys, key=lambda item: int(item.rsplit(b":", 1)[1]))
            number = int(key.rsplit(b":", 1)[1])
            image, timestamp, boxes, distance = self.redis.hmget(
                key, "image", "sim_time", "detections", "distance_meter")
            if not image or timestamp is None:
                continue
            source_t = float(timestamp)
            previous = self.latest.get(uid)
            if previous and (previous["frame_no"], previous["source_sim_time"]) == (number, source_t):
                continue
            frame = {"uid": uid, "frame_no": number, "source_sim_time": source_t,
                     "received_unix_s": time.time(), "image": image,
                     # UE 框元数据仅供离线识别画面中的车辆，不代表已通过遮挡检测。
                     "ue_detections": json.loads(boxes) if boxes else [],
                     "ue_distance_meter": float(distance) if distance else None}
            self.latest[uid] = frame
            # 按源仿真时间限制为每机约 2 Hz，保留约六秒前置画面。
            if previous and source_t - self.last_sample.get(uid, -1e30) < 0.5:
                continue
            self.last_sample[uid] = source_t
            self.frames[uid].append(frame)
            if previous is None or time.monotonic() <= self.until:
                self._save(frame)

    def _loop(self):
        while not self.stop_event.is_set():
            self._drain()
            try:
                self._poll()
            except (redis.RedisError, ValueError) as exc:
                self.stats["poll_errors"] += 1
                self.stats["last_poll_error"] = str(exc)
            self.stop_event.wait(0.2)
        self._drain()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.stats["open_gaps_at_end"] = self.gaps
        self.stats["latest_frames"] = {
            uid: {key: value for key, value in frame.items() if key != "image"}
            for uid, frame in self.latest.items()}
        (self.output / "summary.json").write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
        self.events_file.close()
        self.frames_file.close()
        self.redis.close()


class StudyRenderer:
    """启动本次实验专属 UE；按官方服务协议加载、分配相机和关闭。"""

    def __init__(self, root, output, host, port, log):
        self.root, self.output, self.log = Path(root), Path(output), log
        self.host, self.port = host, port
        self.redis = redis.Redis(host=host, port=port, socket_timeout=2)
        self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe("sim:control", "sim:render_id")
        self.process = None
        self.render_id = None
        self.stream = (self.output / "ue.log").open("wb")
        self.protocol = (self.output / "ue_protocol.jsonl").open("w", encoding="utf-8", buffering=1)

    def _wait(self, predicate, timeout):
        end = time.monotonic() + timeout
        next_notice = time.monotonic() + 10
        while time.monotonic() < end:
            message = self.pubsub.get_message(timeout=0.5)
            if message:
                data = json.loads(message["data"])
                self.protocol.write(json.dumps({"received_unix_s": time.time(), "message": data}) + "\n")
                if predicate(data):
                    return data
            if time.monotonic() >= next_notice:
                self.log("[capture] 等待 UE 服务就绪…")
                next_notice += 10
            if self.process.poll() is not None:
                raise RuntimeError("本次 UE 已退出，请查看 ue.log")
        raise TimeoutError("UE 服务等待超时，请查看 ue_protocol.jsonl")

    def start(self, scenario, uids):
        # Shipping UE 从此文件读配置；只验证和保存快照，绝不改写共享发行包。
        capture_path = self.root / "ue-renderer/Windows/testwl/Content/Config/capture_config.json"
        capture_bytes = capture_path.read_bytes()
        capture = json.loads(capture_bytes.decode("utf-8-sig"))
        endpoint = capture.get("redis", {})
        if (capture.get("render_mode") != "service"
                or endpoint.get("host") != self.host or endpoint.get("port") != self.port
                or capture.get("scenario_override") or capture.get("save_image", {}).get("enabled")):
            raise ValueError("UE 配置需为 service、相同 Redis、空场景覆盖且关闭独立图片写盘；未改写发行包")
        (self.output / "capture_config.json").write_bytes(capture_bytes)
        config = json.loads((self.root / "config/renderers/ue_testwl.json").read_text(encoding="utf-8"))
        exe = config["executable"]
        workdir = self.root / exe["workdir"]
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        self.process = subprocess.Popen(
            [str(workdir / exe["launcher"]), "/Env_MultiBS_Data/Maps/Map_MultiBS.Map_MultiBS", *exe["args"]],
            cwd=workdir, stdout=self.stream, stderr=subprocess.STDOUT, startupinfo=startup)
        online = self._wait(lambda m: m.get("event") == "renderer_online", 180)
        self.render_id = online["render_id"]
        self.redis.set("sim:scenario", json.dumps(scenario))
        self.redis.publish("sim:control", json.dumps({"action": "load_scenario", "render_id": self.render_id}))
        self._wait(lambda m: m.get("render_id") == self.render_id and m.get("state") == "rendering", 90)
        self.redis.publish("sim:render_id", json.dumps({"event": "assign", "render_id": self.render_id,
                                                       "aircraft": list(uids)}))
        self.log(f"[capture] UE 已分配三机相机：{self.render_id}")

    def close(self):
        if self.process and self.process.poll() is None:
            if self.render_id:
                try:
                    self.redis.publish("sim:control", json.dumps({"action": "shutdown", "render_id": self.render_id}))
                except redis.RedisError as exc:
                    self.log(f"[capture] UE 关闭通知失败，继续清理本轮进程：{exc}")
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # 仅清理本次启动的进程树，不终止其它实验或已有 UE。
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pubsub.close()
        self.redis.close()
        self.stream.close()
        self.protocol.close()
