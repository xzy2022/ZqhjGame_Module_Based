# 修改时间：2026-09-13。
# 修改目的：同步三次检测计数修复后的冻结记录和报告口径。
# 修改内容：记录首次计数为一和旧结果失效原因并更正理论冷启动为约零点九秒。
# 修改时间：2026-09-13。
# 修改目的：冻结配置后一次性完成间隔对照、test 和中文交付。
# 修改内容：增加冻结批次入口及全体子集指标、失败区间、发现时延和可追溯报告。
# 修改时间：2026-09-13。
# 修改目的：明确诊断阶段耗时与正式计时的边界。
# 修改内容：更正观察器耗时注释并保留诊断开销标记。
# 修改时间：2026-09-13。
# 修改目的：保存动态发现路线的可复现离线实验和诊断证据。
# 修改内容：封装公共运行评估接口、源时间适配、残差观察、消融统计与视频输出。
"""标签仅由运行后的诊断和公共评估读取，检测器只得到 RGB 与源时间。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from . import common
from .motion import DEFAULT_CONFIG, build_detector


def metrics(manifest, records, split):
    raw = [r | {'candidates': r.get('raw_candidates', [])} for r in records]
    results = {}
    for name, rows in [('stable', records), ('raw', raw)]:
        results[name] = {}
        for label, fov in [('all', None), ('fov30', 30), ('fov48', 48)]:
            results[name][label] = {
                'overall': common.evaluate(manifest, rows, split, fov),
                'registered': common.evaluate(manifest, rows, split, fov, processed_only=True)}
    results['reasons'] = dict(Counter(r['reason'] for r in records))
    results['per_clip'] = {clip['id']: {
        'stable': common.evaluate({'clips': [clip]}, records, split),
        'raw': common.evaluate({'clips': [clip]}, raw, split)}
        for clip in manifest['clips'] if clip['split'] == split}
    results['registration'] = {key: {
        'median': float(np.median(values)), 'p95': float(np.percentile(values, 95))}
        for key in ['inlier_ratio', 'median_error', 'overlap', 'residual_p95', 'residual_fraction', 'raw_count']
        if (values := [r['diagnostics'][key] for r in records if r['processed'] and key in r['diagnostics']])}
    return results


def run(manifest, config, output, split, diagnostics=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    common.write_json(output/'config.json', config)
    # 公共运行器传入 time 字段，因此只在运行视图中将采样网格一并移到源时钟。
    # 返回 records 后恢复相对时间，保证公共事件评估仍使用原 manifest。
    run_manifest = {'clips': []}
    original_frames = {}
    for clip in manifest['clips']:
        offset = clip['frames'][0]['source_time'] - clip['frames'][0]['time']
        run_manifest['clips'].append(clip | {'start': clip['start'] + offset, 'end': clip['end'] + offset,
            'frames': [f | {'time': f['source_time']} for f in clip['frames']]})
        original_frames.update({f['key']: f for f in clip['frames']})
    observations = []
    chosen_clips = iter(c for c in manifest['clips'] if c['split'] == split)

    def factory():
        detector = build_detector(config)
        clip = next(chosen_clips)
        frame_index = {f['source_time']: f for f in clip['frames']}
        snapshots = Counter()

        class Observer:
            def reset(self):
                detector.reset()

            def detect(self, rgb, time):
                result = detector.detect(rgb, time)
                # 诊断模式包含下列观察和落盘开销，正式比较时关闭观察器。
                if diagnostics and result['processed']:
                    frame = frame_index[time]
                    debug = detector.last_debug
                    for label in frame['labels']:
                        x1, y1, x2, y2 = map(int, label['box'])
                        values = debug['residual'][y1:y2, x1:x2]
                        valid = debug['mask'][y1:y2, x1:x2] > 0
                        if values.size:
                            observations.append({'key': frame['key'], 'clip': clip['id'], 'time': frame['time'],
                                'moving': label['moving'], 'object': label['id'], 'box': label['box'],
                                'valid_fraction': float(np.mean(valid)), 'residual_mean': float(values.mean()),
                                'residual_p95': float(np.percentile(values, 95)),
                                'active_fraction': float(np.mean(debug['binary'][y1:y2, x1:x2] > 0)),
                                'raw_max_iou': max((common.iou(c['box'], label['box']) for c in result['raw_candidates']), default=0.)})
                    kind = 'moving' if any(g['moving'] for g in frame['labels']) else ('vehicle' if frame['labels'] else 'background')
                    if snapshots[kind] < 3:
                        snapshots[kind] += 1
                        save_diagnostic(output/'diagnostics', clip['id'], frame, rgb, result, debug)
                return result
        return Observer() if diagnostics else detector

    # 诊断模式的耗时含图像保存，仅用于观察；正式间隔比较与最终轮不开诊断。
    records = common.run_detector(run_manifest, factory, config['interval'], split)
    for record in records:
        record['time'] = original_frames[record['key']]['time']
    common.save_records(output/'records.jsonl', records)
    common.save_records(output/'raw_records.jsonl', [r | {'candidates': r['raw_candidates']} for r in records])
    stats = metrics(manifest, records, split)
    stats['timing_includes_diagnostic_io'] = diagnostics
    common.write_json(output/'metrics.json', stats)
    if diagnostics:
        common.save_records(output/'vehicle_residual_observations.jsonl', observations)
    short_keys = ['frames', 'moving_recall', 'fp_per_frame', 'moving_discovery_1s', 'mean_ms']
    print(json.dumps({'output': str(output), 'split': split, 'reasons': stats['reasons'],
        **{name: {k: stats[name]['all']['overall'][k] for k in short_keys} for name in ['stable', 'raw']}}, ensure_ascii=False), flush=True)
    return records, stats


def save_diagnostic(output, clip, frame, rgb, result, debug):
    output.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panels = [bgr, cv2.cvtColor(debug['warped'], cv2.COLOR_GRAY2BGR),
              cv2.applyColorMap(np.uint8(np.minimum(debug['residual'] * 4, 255)), cv2.COLORMAP_TURBO),
              cv2.cvtColor(debug['binary'], cv2.COLOR_GRAY2BGR)]
    titles = ['current / raw orange / GT green', 'previous warped by H', 'absolute residual x4', 'overlap + edge mask + components']
    for panel, title in zip(panels, titles):
        for g in frame['labels']:
            x1, y1, x2, y2 = map(int, g['box'])
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(panel, title, (10, 25), 0, .65, (255, 255, 255), 1)
    for c in result['raw_candidates']:
        x1, y1, x2, y2 = map(int, c['box'])
        cv2.rectangle(panels[0], (x1, y1), (x2, y2), (0, 140, 255), 1)
    name = f'{clip}_{frame["time"]:.3f}'
    montage = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    cv2.imwrite(str(output/(name+'.jpg')), montage)
    for index, g in enumerate(frame['labels']):
        x1, y1, x2, y2 = map(int, g['box'])
        x1, y1 = max(0, x1-25), max(0, y1-25)
        x2, y2 = min(rgb.shape[1], x2+25), min(rgb.shape[0], y2+25)
        crop = np.hstack([p[y1:y2, x1:x2] for p in panels])
        cv2.imwrite(str(output/(name+f'_vehicle_{index}.png')), cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


def failure_intervals(manifest, records):
    intervals = []
    for clip in manifest['clips']:
        rows = [r for r in records if r['clip'] == clip['id']]
        pending = []
        for row in rows + [None]:
            if row is not None and not row['processed']:
                pending.append(row)
            elif pending:
                end = row['time'] if row is not None else clip['end']
                intervals.append({'clip': clip['id'], 'start': pending[0]['time'],
                    'last_skipped_time': pending[-1]['time'], 'end': end,
                    'duration_to_next_sample_or_clip_end_s': end - pending[0]['time'],
                    'frames': len(pending), 'reasons': dict(Counter(r['reason'] for r in pending)),
                    'recovered': row is not None})
                pending = []
    return intervals


def discovery_latencies(manifest, records, stats, split):
    lookup = {r['key']: r for r in records}
    events = []
    for event in stats['stable']['all']['overall']['event_details']:
        clip = next(c for c in manifest['clips'] if c['id'] == event['clip'])
        entry = dict(event)
        for name in ['raw', 'stable']:
            times = []
            for frame in clip['frames']:
                if not event['start'] <= frame['time'] <= event['start'] + event['visible_duration'] + 1e-6:
                    continue
                row = lookup.get(frame['key'])
                if row is None:
                    continue
                candidates = row['raw_candidates'] if name == 'raw' else row['candidates']
                pairs = common.match(candidates, frame['labels'])
                if any(frame['labels'][j]['id'] == event['object'] for j in pairs.values()):
                    times.append(frame['time'])
            entry[name + '_first_latency_s'] = min(times) - event['start'] if times else None
        events.append(entry)
    return events


def frozen_batch(manifest, manifest_path, config, output, videos):
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config) | {'interval': .3}
    common.write_json(output/'config.json', config)
    freeze = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'selected_round': 2,
        'selection': '沿用计数修复前已冻结的第二轮全部参数，未依据修复后的 tune/test 重新选参；主对比固定 0.3s。',
        'counting_fix': '首次候选 steps=1，第三次连续残差检测即可输出；旧 steps=0 结果保留在 superseded_counting_bug。',
        'supersedes': 'superseded_counting_bug',
        'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256((output/'config.json').read_bytes()).hexdigest(),
        'detector_sha256': hashlib.sha256(Path(__file__).with_name('motion.py').read_bytes()).hexdigest(),
        'common_sha256': hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'cv2': cv2.__version__, 'threads': 2, 'test_detector_runs': 1}
    common.write_json(output/'freeze.json', freeze)
    sensitivity = {}
    selected_records = None
    for interval in [.2, .3, .5]:
        records, stats = run(manifest, config | {'interval': interval}, output/f'tune/interval_{interval:.1f}', 'tune')
        sensitivity[str(interval)] = stats
        if interval == .3:
            selected_records = records
    # 只有这里执行一次冻结后的 test，不将 test 结果用于配置选择。
    test_records, test_stats = run(manifest, config, output/'test', 'test')
    common.save_records(output/'final_records.jsonl', selected_records + test_records)
    common.save_records(output/'raw_records.jsonl', [r | {'candidates': r['raw_candidates']} for r in selected_records + test_records])
    rounds = {path.parent.name: common.read_json(path) for path in sorted((output/'tune').glob('round*/metrics.json'))}
    summary = {'freeze': freeze, 'interval_sensitivity_tune': sensitivity, 'tune_rounds': rounds,
        'tune': sensitivity['0.3'], 'test': test_stats,
        'failure_intervals': failure_intervals(manifest, selected_records + test_records),
        'discovery_latencies': {split: discovery_latencies(manifest, rows, stats, split)
            for split, rows, stats in [('tune', selected_records, sensitivity['0.3']), ('test', test_records, test_stats)]},
        'final_records_contract': '0.3s tune + test，共用原 manifest key；test/records.jsonl 为仅 test 子集。',
        'timing_contract': 'common.run_detector 检测调用耗时；不含图像解码和视频生成；raw 消融复用完整流程耗时；诊断轮包含额外观察落盘。'}
    common.write_json(output/'summary.json', summary)
    common.write_json(output/'tune/interval_summary.json', sensitivity)
    common.write_json(output/'tune/round_summary.json', rounds)
    write_report(output, summary)
    if videos:
        for split, rows in [('tune', selected_records), ('test', test_records)]:
            common.write_videos(manifest, rows, output/f'videos/{split}/stable', 'motion stable', .3, split)
            common.write_videos(manifest, [r | {'candidates': r['raw_candidates']} for r in rows],
                                output/f'videos/{split}/raw', 'motion raw', .3, split)


def write_report(output, summary):
    def fmt(value):
        return '无分母' if value is None else f'{value:.3f}'
    lines = ['# 动态车辆发现离线可行性报告', '',
        '检测器仅接收 RGB 图像与源 time；不读取标签、姿态，不作真假分类、不控制、不启动仿真。标签的 moving 字段完全沿用公共基础 b777421 的 manifest。', '',
        '流程：ORB 比率匹配 → RANSAC 背景单应 → 公共区域腐蚀及共同边缘去除 → 带 1px 容差的灰度残差 → 连通域 → 上帧候选经 H 映射后按合理残余位移关联 → 连续三次残差检测稳定输出。', '',
        '每个片段 reset；首帧跳过；配准失败更新图像参考并清空候选历史，后续可恢复。首次候选 steps=1，第三次连续候选达到 stable_steps=3 即可输出。初帧初始化后，在 0.3/0.6/0.9s 各产生候选时，第三次即可输出；理论冷启动约 0.9s，不再由计数错误强制拖到 1.2s。', '',
        '计数修复仅将首次 steps 从 0 改为 1，所有冻结参数完全保留。旧输出标记 superseded_counting_bug；当前各轮重算仅校正统计口径，没有额外调参。源码哈希在修复后重新冻结，test 按相同配置重新运行一次；旧版 test 只作为失效历史保留。纯关联合成测试验证第三次才输出、失败后清零重计。', '',
        '已完成初始诊断和最多两轮调整。round0 发现共同边缘掩膜切碎车辆、树石纹理产生碎片；round1 加入 1px 容差并缩窄边缘删除，提升 raw 动态召回但误报增多；round2 用原始残差估计噪声阈值，设置 3px/0.3s 最小位移和速度预测关联，缓解旧位置拖影抢关联。', '',
        '参数早于首次 test 冻结为 0.3s；计数修复后仅更新源代码哈希再次冻结。0.2/0.5s 仅在 tune 上做敏感性对照，修复版 test 检测只运行一轮。没有按新旧 test 调参。', '',
        '## 主对比', '',
        'IoU=0.25。全体包括采样后的首帧和所有配准失败帧；成功配准子集只保留 processed=true。整体分母为公共采样网格上的全部记录，不是 manifest 的每一张原始图。moving 召回单独计算；误报按所有车辆标签匹配，命中静止车辆不算误报。', '',
        '| 划分 | 输出 | 范围 | 帧数 | moving TP/GT | 动态召回 | 全车辆召回 | FP/帧 | 动态1s发现 | 平均ms |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ['tune', 'test']:
        for name in ['raw', 'stable']:
            for scope in ['overall', 'registered']:
                m = summary[split][name]['all'][scope]
                lines.append(f"| {split} | {name} | {scope} | {m['frames']} | {m['moving_tp']}/{m['moving_gt']} | {fmt(m['moving_recall'])} | {fmt(m['recall'])} | {fmt(m['fp_per_frame'])} | {m['moving_found_1s']}/{m['moving_events']} | {fmt(m['mean_ms'])} |")
    lines += ['', '检测计时不含图像解码、视频编码；raw 是输出消融，耗时复用完整检测流程，不能解释成独立 raw 算法耗时。诊断轮额外包含 PNG/JPEG 与车辆邻域统计，不能用于速度主对比。OpenCV 固定两线程，未安装依赖；并行任务负载可能影响墙钟耗时。', '',
              '## Tune 各轮及间隔', '', '| 轮次/间隔 | raw动态召回 | stable动态召回 | raw FP/帧 | stable FP/帧 | 配准成功率 |', '|---|---:|---:|---:|---:|---:|']
    for name, stats in list(summary['tune_rounds'].items()) + [(f'冻结后 {interval}s', s) for interval, s in summary['interval_sensitivity_tune'].items()]:
        raw, stable = stats['raw']['all']['overall'], stats['stable']['all']['overall']
        lines.append(f"| {name} | {fmt(raw['moving_recall'])} | {fmt(stable['moving_recall'])} | {fmt(raw['fp_per_frame'])} | {fmt(stable['fp_per_frame'])} | {fmt(stable['processed_ratio'])} |")
    lines += ['', '## 按 FOV 的 test 主对比', '', '| FOV | 输出 | 范围 | moving TP/GT | FP/帧 |', '|---|---|---|---:|---:|']
    for fov in ['fov30', 'fov48']:
        for name in ['raw', 'stable']:
            for scope in ['overall', 'registered']:
                m = summary['test'][name][fov][scope]
                lines.append(f"| {fov} | {name} | {scope} | {m['moving_tp']}/{m['moving_gt']} | {fmt(m['fp_per_frame'])} |")
    lines += ['', '## 跳过、恢复与发现时延', '',
              '区间结束取下一成功采样时刻或片段终点，时长表示采样覆盖区间；精确最后失败时刻另存 summary.json。相邻失败按连续区间合并，原因计数保留。', '',
              '| 片段 | 开始s | 结束s | 覆盖时长s | 跳过帧 | 原因 | 后续恢复 |', '|---|---:|---:|---:|---:|---|---|']
    for item in summary['failure_intervals']:
        lines.append(f"| {item['clip']} | {item['start']:.3f} | {item['end']:.3f} | {item['duration_to_next_sample_or_clip_end_s']:.3f} | {item['frames']} | {json.dumps(item['reasons'])} | {item['recovered']} |")
    lines += ['', '| 划分/片段 | moving目标 | 可见起点s | 可见时长s | raw首次时延s | stable首次时延s |', '|---|---|---:|---:|---:|---:|']
    for split, events in summary['discovery_latencies'].items():
        for event in events:
            if event['moving']:
                raw = event['raw_first_latency_s']; stable = event['stable_first_latency_s']
                lines.append(f"| {split}/{event['clip']} | {event['object']} | {event['start']:.3f} | {event['visible_duration']:.3f} | {'未发现' if raw is None else fmt(raw)} | {'未发现' if stable is None else fmt(stable)} |")
    lines += ['', '## 结论与边界', '',
        '图像背景配准后的车辆残差有离线证据，连续过滤能够大幅减少纹理碎片，但小目标碎裂、差分拖影、进出公共视野和背景非平面结构仍会造成漏检或假运动。成功配准子集也不能掩盖这些候选与连续性损失。三次连续检测的冷启动和重新积累时间仍影响一秒发现率，但不会因首次计零而必然阻止入口一秒发现；这条路线的有限可行性不能等同于完整动态车辆发现能力。', '',
        '车辆刚进入画面时可能尚未进入两帧公共区域。诊断中的高原始残差并不表示可用车辆候选；被公共区域 mask 排除的像素不参与候选生成。单应只描述全局平面背景，树木、岩石的深度差和渲染变化可能留下残差。', '',
        '## 交付与复现', '',
        '- config.json：冻结配置，可直接传给 motion.build_detector(config)。',
        '- final_records.jsonl / raw_records.jsonl：0.3s tune+test 稳定输出及稳定前消融；每条保留 raw_candidates。',
        '- test/records.jsonl、test/raw_records.jsonl、test/metrics.json：唯一 test 检测轮。',
        '- tune/round*_diagnostics：各轮配置、指标、配准/差分图和车辆邻域残差 JSONL。',
        '- tune/interval_*、interval_summary.json、round_summary.json：冻结后 tune 间隔对照和各轮统计。',
        '- summary.json：全体/成功子集/FOV/片段指标、失败区间和发现时延；freeze.json：test 之前写入的配置与源文件哈希。',
        '- videos/{tune,test}/{stable,raw}：绿色标签框，橙色候选框。', '',
        '重新运行冻结配置时使用新的输出目录：', '', '```powershell',
        "Set-Location 'D:/Workspace/00_MyRepo/red_m_competiton/hf2026-vehicle-motion-poc'",
        "& 'D:/Workspace/00_MyRepo/red_m_competiton/hf2026-sim-windows/ZqhjGame/.venv-learning/Scripts/python.exe' -m competition.user_algorithms.coop_decoy.vehicle_discovery.motion_run `",
        f"  --manifest '{output.parent.as_posix()}/manifest.json' `",
        f"  --config '{output.as_posix()}/config.json' `",
        f"  --output '{output.as_posix()}-rerun' --frozen-batch --videos", '```', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='动态车辆离线发现；不启动仿真。')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--interval', type=float)
    parser.add_argument('--split', choices=['tune', 'test'], default='tune')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--videos', action='store_true')
    parser.add_argument('--frozen-batch', action='store_true', help='冻结 0.3s，运行 tune 三间隔和一轮 test 并汇总。')
    args = parser.parse_args()
    config = DEFAULT_CONFIG | (common.read_json(args.config) if args.config else {})
    if args.interval is not None:
        config['interval'] = args.interval
    manifest = common.read_json(args.manifest)
    output = Path(args.output)
    if args.frozen_batch:
        frozen_batch(manifest, args.manifest, config, output, args.videos)
        return
    records, stats = run(manifest, config, output, args.split, args.diagnostics)
    common.write_json(output/'provenance.json', {
        'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256((output/'config.json').read_bytes()).hexdigest(),
        'detector_sha256': hashlib.sha256(Path(__file__).with_name('motion.py').read_bytes()).hexdigest(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'cv2': cv2.__version__, 'threads': cv2.getNumThreads(), 'split': args.split})
    if args.videos:
        common.write_videos(manifest, records, output/'videos_stable', 'motion stable', config['interval'], args.split)
        common.write_videos(manifest, [r | {'candidates': r['raw_candidates']} for r in records],
                            output/'videos_raw', 'motion raw', config['interval'], args.split)


if __name__ == '__main__':
    main()
