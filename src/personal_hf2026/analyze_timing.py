# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
"""事后读取实验记录，对比本地计时和裁判计时，不连接仿真。"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from competition.baselines.coop_distributed import _haversine_m
from .coop_clock import CoopClock


def replay(rows, tolerance, distance=10.0):
    # 用双机观测时间配对；不把发送瞬间的漏检布尔值延长到下一条消息。
    clock = CoopClock()
    clock.PAIR_TOLERANCE_S = tolerance
    clock.PAIR_DISTANCE_M = distance
    done = None
    first_pair = None
    for row in rows:
        now = row["t"]
        # 日志保存决策后的状态；进入 DONE 的这一帧仍执行过计时，必须回放。
        finishing = (row["state"] == "DONE"
                     and row.get("summary", {}).get("done_at_s") == now)
        if row["state"] != "TRACK" and not finishing:
            continue
        det = row["self"]["detection"]
        pos = None
        if (det["detected"] and det["target_lat"] is not None and row["candidate"]
                and _haversine_m(det["target_lat"], det["target_lon"], *row["candidate"]) < 250):
            pos = (det["target_lat"], det["target_lon"])
        clock.observe(now, pos, [SimpleNamespace(**m) for m in row["inbox"]])
        if clock.updated and first_pair is None:
            first_pair = now
        if clock.updated and clock.seconds >= 22.0 and done is None:
            done = now
    return {"tolerance": tolerance, "distance": distance, "first_pair": first_pair, "max_seconds": clock.peak_seconds,
            "done_at": done, "resets": clock.resets}


def analyze(folder):
    rows = [json.loads(s) for s in (folder / "observations.jsonl").read_text().splitlines()]
    a = [r for r in rows if r["uid"] == "20001"]
    judge = [json.loads(s) for s in (folder / "judge.jsonl").read_text().splitlines()]
    completed = {}
    # 统一按真目标的 K=2、两秒中断规则回放所有身份，专门量化诱饵分支差异。
    uniform = {}
    uniform_completed = {}
    previous_t = judge[0]["t"]
    for row in judge:
        dt = row["t"] - previous_t
        previous_t = row["t"]
        ids = [m["target_uid"] or m["decoy_uid"] for m in row["matches"].values()]
        for uid in set(row["targets"]) | set(row["decoys"]):
            state = uniform.setdefault(uid, {"seconds": 0.0, "gap": 0.0, "active": False})
            if ids.count(uid) >= 2:
                state["seconds"] += (state["gap"] if state["active"] else 0.0) + dt
                state["gap"] = 0.0
                state["active"] = True
            else:
                state["gap"] += dt
                if state["active"] and state["gap"] > 2.0:
                    state.update(seconds=0.0, gap=0.0, active=False)
            if state["seconds"] >= 20.0:
                uniform_completed.setdefault(uid, row["t"])
        for group, flag in (("targets", "destroyed"), ("decoys", "identified")):
            for uid, state in row[group].items():
                if state[flag]:
                    completed.setdefault(uid, row["t"])
    agents = {}
    for uid in ("20001", "20002"):
        rr = [r for r in rows if r["uid"] == uid]
        first = {}
        for r in rr:
            first.setdefault(r["state"], r["t"])
        agents[uid] = {"first_state": first, "final": rr[-1]["summary"],
                       "max_seconds": max(r["coop"] for r in rr),
                       "last_detection_age": rr[-1]["t"] - rr[-1]["last_seen"],
                       "final_candidate": rr[-1]["candidate"],
                       "final_distance": rr[-1]["distance_to_candidate"],
                       "min_distance": min((r["distance_to_candidate"] for r in rr
                                            if r["distance_to_candidate"] is not None), default=None)}
    replays = [replay(a, tol, distance) for tol in (0.15, 0.3, 0.5) for distance in (10.0, 250.0)]
    for r in replays:
        r["judge_already_completed"] = (any(t <= r["done_at"] for t in completed.values())
                                          if r["done_at"] is not None else None)
        r["uniform_rule_already_completed"] = (
            any(t <= r["done_at"] for t in uniform_completed.values())
            if r["done_at"] is not None else None)
    result = {"run": folder.name, "end": judge[-1]["t"], "agents": agents,
              "judge_completed": completed,
              "uniform_k2_grace2_completion": uniform_completed,
              "timestamp_replays": replays}
    (folder / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--compact-trace")
    args = parser.parse_args()
    root = Path(args.root)
    results = [analyze(p) for p in sorted(root.iterdir())
               if p.is_dir() and (p / "summary.json").exists()]
    (root / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        print(json.dumps(r))
    if args.compact_trace:
        # 旧诊断只存最后检测时间；以下是重建回放，不冒充新一轮仿真。
        source = Path(args.compact_trace)
        raw = source.read_bytes()
        (root / "historical_compact_input.jsonl").write_bytes(raw)
        old = [json.loads(s) for s in raw.decode().splitlines()]
        old = [r for r in old if r["uid"] == "20001"]
        converted = []
        for i, r in enumerate(old):
            peer = r["peer"]
            pos = r["candidate"]
            msgs = [] if peer is None else [{"sender_uid": "20002",
                "payload": f"1,{i},{peer[0]:.1f},{int(peer[1])},{peer[2][0]},{peer[2][1]}"}]
            converted.append({"t": r["t"], "state": r["state"], "candidate": pos,
                              "self": {"detection": {"detected": abs(r["t"] - r["last_seen"]) < 0.051,
                                                      "target_lat": pos[0] if pos else None,
                                                      "target_lon": pos[1] if pos else None}},
                              "inbox": msgs})
        result = {"source": str(source.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                  "kind": "approximate_reconstruction_from_compact_trace",
                  "original_peak_seconds": max(r["coop"] for r in old),
                  "replays": [replay(converted, tol) for tol in (0.15, 0.3, 0.5)]}
        (root / "historical_replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
