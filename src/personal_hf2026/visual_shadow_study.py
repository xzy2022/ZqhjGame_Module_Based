# 修改时间：2026-09-14
# 修改目的：确保相对运行底座路径在切换工作目录后仍能正确定位 UE。
# 修改内容：在视觉 Runner 构造时将运行底座和模型路径转换为绝对路径。
# 修改时间：2026-09-14
# 修改目的：让个人实验脱离官方仓库的后续修改并支持独立运行。
# 修改内容：统一模块、SDK、运行资源及输出路径并保留实验行为。
# 修改时间：2026-09-13（采集收尾）
# 修改目的：让专用采集入口识别相机保存失败，并能关闭尚未启动的相机资源。
# 修改内容：记录相机异常计数，并使停止接口可重复调用且支持启动失败收尾。
# 修改时间：2026-09-13（FOV 对照采集）
# 修改目的：允许单次采集实验覆盖 V2 的 FOV，而不改变默认基线。
# 修改内容：增加受比赛范围约束的 --fov 参数，并记录请求值和实际值。
# 修改时间：2026-09-13（按需数据采集）
# 修改目的：允许仅采集图像与状态而不启动视觉分类。
# 修改内容：增加采集开关、唯一新帧保存、首次读取时间和离线索引生成。
# 修改时间：2026-09-13
# 修改目的：用真实 RGB 和帧源时间验证视觉旁路，保持原版控制和评分。
# 修改内容：新增专用相机缓存、V2 实验入口、全拍 V1 动作对照和模型快照。
"""个人 V2 视觉旁路实验；只运行一次指定时长，默认静态诱饵。"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .paths import PROJECT_ROOT, SIM_ROOT, RUNTIME_ROOT, OUTPUT_ROOT, SCENARIO_ROOT
import threading
import time

import redis

from competition.sdk.core.runner import ScenarioConfig
from competition.sdk.core.perception import DetectionResolver
from .control_test_runner import MultiTargetIdealDetector
from .dropout_capture import StudyRenderer
from .personal_v1 import PersonalV1Agent
from .personal_v2 import PersonalV2Agent, VisualShadow
from .timing_study import StudyRunner


ROOT = PROJECT_ROOT


class TimedPhotoCache:
    """同一帧原子读取图片与源时间；真实对象元数据只写审计文件。"""
    def __init__(self, uids, output, host, port, collect_dataset=False):
        self.uids, self.output = uids, Path(output)
        self.redis = redis.Redis(host=host, port=port, socket_timeout=2)
        self.latest, self.delivered = {}, {}
        self.stop_event = threading.Event()
        self.error_count = 0
        self.closed = False
        # 拒绝以前运行残留的帧，只有本次帧记录变化后才开始交付。
        self.baseline = {uid: self._read(uid) for uid in uids}
        self.activated = set()
        self.dataset = None
        if collect_dataset:
            from .capture_dataset import DatasetRecorder
            self.dataset = DatasetRecorder(self.output)
        self.audit = (self.output / "frame_audit.jsonl").open("w", encoding="utf-8", buffering=1)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _read(self, uid):
        keys = list(self.redis.scan_iter(match=f"sync_camera:{uid}:frame:*", count=100))
        if not keys:
            return None
        key = max(keys, key=lambda k: int(k.rsplit(b":", 1)[1]))
        image, source, boxes = self.redis.hmget(key, "image", "sim_time", "detections")
        if not image or source is None:
            return None
        return {"frame_no": int(key.rsplit(b":", 1)[1]), "source_sim_time": float(source),
                "image": image, "audit_boxes": json.loads(boxes) if boxes else [],
                "received_unix_s": time.time()}

    def start(self):
        self.thread.start()

    def _loop(self):
        last_logged = {}
        while not self.stop_event.is_set():
            for uid in self.uids:
                try:
                    raw = self._read(uid)
                    if raw is None:
                        continue
                    signature = (raw["frame_no"], raw["source_sim_time"])
                    old = self.baseline[uid]
                    if uid not in self.activated:
                        if old and signature == (old["frame_no"], old["source_sim_time"]):
                            continue
                        self.activated.add(uid)
                    self.latest[uid] = {k: raw[k] for k in ("frame_no", "source_sim_time", "image")}
                    if signature != last_logged.get(uid):
                        if self.dataset:
                            self.dataset.record(uid, raw)
                        self.audit.write(json.dumps({"uid": uid, "frame_no": raw["frame_no"],
                            "source_sim_time": raw["source_sim_time"], "ue_detections": raw["audit_boxes"]}) + "\n")
                        last_logged[uid] = signature
                except Exception as exc:
                    self.error_count += 1
                    self.audit.write(json.dumps({"uid": uid, "error": repr(exc)}) + "\n")
            self.stop_event.wait(0.05)

    def get(self, uid):
        packet = self.latest.get(uid)
        self.delivered[uid] = packet
        return packet["image"] if packet else None

    def stop(self):
        if self.closed:
            return
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                # 不在父 runner 的 finally 中抛出，否则会跳过原生引擎清理。
                self.error_count += 1
                return
        self.audit.close()
        if self.dataset:
            self.dataset.close()
        self.redis.close()
        self.closed = True


class VisualStudyRunner(StudyRunner):
    def __init__(self, cfg, output, runtime_root, weights, log, collect_dataset=False):
        super().__init__(cfg, PersonalV2Agent, output, log)
        self.runtime_root, self.weights = Path(runtime_root).resolve(), Path(weights).resolve()
        self.shadows, self.twins = {}, {}
        self.camera = None
        self.collect_dataset = collect_dataset
        self.delivered_frames = set()
        self.delivery_log = None
        if collect_dataset:
            (output / "dataset").mkdir(exist_ok=True)
            self.delivery_log = (output / "dataset/deliveries.jsonl").open("w", encoding="utf-8", buffering=1)
        self.control_check = {"ticks": 0, "mismatches": 0}
        self.control_log = (output / "control_equivalence.jsonl").open("w", encoding="utf-8", buffering=1)

    def _build_perception(self, uids):
        self.camera = TimedPhotoCache(uids, self.output, self.cfg.redis_host, self.cfg.redis_port,
                                     self.collect_dataset)
        self.renderer = StudyRenderer(self.runtime_root, self.output, self.cfg.redis_host, self.cfg.redis_port, self.log)
        self.renderer.start(self._scenario_cfg, uids)
        self.camera.start()
        return self.camera, DetectionResolver(default_detector=MultiTargetIdealDetector())

    def make_agent_for(self, entity_type, entity_uid, world_state):
        agent = super().make_agent_for(entity_type, entity_uid, world_state)
        twin = PersonalV1Agent(my_uid=entity_uid)
        twin.configure(self.agent_config())
        twin.reset()
        self.twins[entity_uid] = twin
        if not self.collect_dataset:
            shadow = VisualShadow(entity_uid, self.output, self.weights)
            agent.shadow = self.shadows[entity_uid] = shadow
        recorded = agent.decide

        def decide(obs, dt):
            packet = self.camera.delivered.get(entity_uid) if self.camera else None
            if packet and obs.briefing.score_view is not None:
                # 原始世界时钟与局内相对时间的偏移仅在实验入口换算。
                offset = self.world_times[entity_uid] - obs.briefing.score_view.sim_time
                agent.photo_packet = {"frame_no": packet["frame_no"],
                    "source_sim_time": packet["source_sim_time"], "source_t": packet["source_sim_time"] - offset}
                signature = (entity_uid, packet["frame_no"], packet["source_sim_time"])
                if self.delivery_log and signature not in self.delivered_frames:
                    self.delivery_log.write(json.dumps(dict(uid=entity_uid, **agent.photo_packet,
                        observe_t=obs.briefing.score_view.sim_time,
                        observed_unix_s=time.time(),
                        frame_age_s=self.world_times[entity_uid] - packet["source_sim_time"])) + "\n")
                    self.delivered_frames.add(signature)
            else:
                agent.photo_packet = None
            expected = twin.decide(obs, dt)
            actual = recorded(obs, dt)
            self.control_check["ticks"] += 1
            if [asdict(x) for x in expected] != [asdict(x) for x in actual]:
                self.control_check["mismatches"] += 1
                self.control_log.write(json.dumps({"uid": entity_uid, "t": agent._t,
                    "expected": [asdict(x) for x in expected], "actual": [asdict(x) for x in actual]}) + "\n")
            return actual

        agent.decide = decide
        return agent

    def should_finish(self, agents):
        super().should_finish(agents)
        # 用户要求完整 200 秒；旁路实验即使 V1 提前完成也观察到指定时长。
        return False

    def close(self):
        for shadow in self.shadows.values():
            shadow.close()
        if self.renderer:
            self.renderer.close()
        self.trace.close()
        self.judge.close()
        self.control_log.close()
        if self.delivery_log:
            self.delivery_log.close()
        (self.output / "control_equivalence.json").write_text(json.dumps(self.control_check, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", default=str(RUNTIME_ROOT))
    parser.add_argument("--layout", default=str(ROOT / "configs/scenarios/coop_decoy/static-decoys.json"))
    parser.add_argument("--duration", type=float, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collect-dataset", action="store_true",
                        help="保存收到的唯一新帧及状态索引；只采集，不运行视觉分类")
    parser.add_argument("--fov", type=float,
                        help="仅本次进程覆盖 V2 搜索与协同 FOV；不传时保持基线默认值")
    args = parser.parse_args()
    if args.fov is not None:
        if not 5.0 <= args.fov <= 50.0:
            parser.error("--fov 必须位于比赛允许的 5～50 度范围")
        PersonalV1Agent.SEARCH_FOV_DEG = float(args.fov)
    if not args.collect_dataset:
        import torch
        import cv2
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    weights = PROJECT_ROOT / "assets/personal_v2/vision.pt"
    snapshots = output / "source"
    snapshots.mkdir()
    hashes = {}
    for path in Path(__file__).parent.glob("*.py"):
        data = path.read_bytes()
        (snapshots / path.name).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    (output / "metadata.json").write_text(json.dumps({"base_commit": "9823b01e04f734790525df9dd3854dea06ea1e60",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "mode": "v2_dataset_only" if args.collect_dataset else "v1_control_visual_shadow",
        "collect_dataset": args.collect_dataset, "visual_inference": not args.collect_dataset,
        "requested_fov_deg": args.fov,
        "effective_search_fov_deg": PersonalV1Agent.SEARCH_FOV_DEG,
        "duration": args.duration, "seed": args.seed,
        "runtime_root": args.runtime_root, "layout": args.layout, "sources": hashes,
        "vision_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "geometry": "horizontal_fov_heading_plus_pan_zero_roll_bearing_only",
        "frame_time": "source_sim_time_not_verified_exposure", "height_input": False,
        "argv": os.sys.argv}, indent=2), encoding="utf-8")
    os.environ["OPENSIM_SIM_STDERR"] = str(output / "engine.log")
    cfg = ScenarioConfig("coop_decoy", args.layout, args.duration, output_dir=str(output),
        sim_binary=str(Path(args.runtime_root) / "opensim-sim.exe"), start_sim_flag=True,
        photo_mode="on", seed=args.seed)
    with (output / "run.log").open("w", encoding="utf-8", buffering=1) as stream:
        def log(message):
            stream.write(str(message) + "\n")
            print(message, flush=True)
        runner = VisualStudyRunner(cfg, output, args.runtime_root, weights, log, args.collect_dataset)
        try:
            result = runner.run()
            (output / "summary.json").write_text(json.dumps({"v1": runner.v1_summary,
                "n_destroyed": result.get("n_destroyed", 0), "error": result.get("error")}, indent=2), encoding="utf-8")
            print(json.dumps(runner.control_check), flush=True)
            if result.get("error"):
                raise RuntimeError(result["error"])
        finally:
            runner.close()
        if args.collect_dataset:
            from .capture_dataset import build_index
            log(json.dumps(build_index(output), ensure_ascii=False))


if __name__ == "__main__":
    main()
