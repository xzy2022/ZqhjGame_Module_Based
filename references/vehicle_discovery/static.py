# 修改时间：2026-09-13。
# 修改目的：补齐交付报告与统一阶段产物而保持冻结检测参数不变。
# 修改内容：保存 A1 规范化调参记录并加入分轮统计、事件计数与可复制重跑命令。
# 修改时间：2026-09-13。
# 修改目的：依据用户澄清优先修正候选截断并避免在低召回上优化分类器。
# 修改内容：第二轮仅增加候选上限，记录中止训练，改为改善后一次固定分类验证并解释事件分母。
# 修改时间：2026-09-13。
# 修改目的：在候选召回通过门槛后完成第二轮二分类实验与冻结测试流程。
# 修改内容：复用四层小卷积、构建仅 tune 训练数据并保存模型统计与一次性测试报告。
# 修改时间：2026-09-13。
# 修改目的：在独立分支验证仅输入原始 RGB 的静态车辆发现候选。
# 修改内容：提供原始候选基线、放宽轮廓与 NMS、统一评估及可重跑接口。
"""静态车辆发现离线实验；检测器不接收标签、位姿或目标运动信息。"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import cv2
import numpy as np
import torch

from . import common
from ..visual_appearance import VehicleAppearance, vehicle_patch, vehicle_proposals

cv2.setNumThreads(2)
torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    # 汇总进程可能已经初始化交互线程，算子内部线程仍固定为二。
    pass

DEFAULT_MANIFEST = 'D:/Workspace/00_MyRepo/red_m_competiton/output/personal_v2/vehicle-discovery/20260913-204344/manifest.json'
A0 = {'variant': 'A0', 'interval': .3, 'threads': 2}
A1 = dict(A0, variant='A1', thresholds=[35, 45, 55, 65, 75, 85],
          adaptive_block=31, adaptive_c=12, gray_ceiling=110,
          area_min=4., area_max=3000., short_min=1.5, long_min=4., long_max=110.,
          ratio_min=1., ratio_max=7., extent_min=.25, solidity_min=.4,
          padding=.12, nms_iou=.65, max_candidates=512)
A1_ROUND2 = dict(A1, max_candidates=2048)


def nms(candidates, threshold, limit):
    """在原图坐标上按几何分数做非极大值抑制。"""
    if not candidates:
        return []
    boxes = np.asarray([c['box'] for c in candidates], np.float32)
    scores = np.asarray([c['score'] for c in candidates])
    areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    order = np.argsort(-scores, kind='stable')
    kept = []
    while len(order) and len(kept) < limit:
        i = int(order[0]); kept.append(candidates[i]); rest = order[1:]
        low = np.maximum(boxes[i, :2], boxes[rest, :2])
        high = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        size = np.maximum(0., high-low)
        inter = size[:, 0]*size[:, 1]
        overlap = inter/np.maximum(1e-9, areas[i]+areas[rest]-inter)
        order = rest[overlap <= threshold]
    return kept


class StaticDetector:
    def __init__(self, config):
        self.config = dict(config)
        self.model = None
        if config['variant'] == 'A2':
            self.model = binary_model()
            self.model.load_state_dict(torch.load(config['model_path'], map_location='cpu', weights_only=True))
            self.model.eval()

    def reset(self):
        pass

    def detect(self, rgb, time):
        cfg = self.config
        if cfg['variant'] == 'A0':
            return [{'box': list(map(float, b)), 'score': 1.} for b in vehicle_proposals(rgb)]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, cfg['adaptive_block'], cfg['adaptive_c'])
        mask[gray > cfg['gray_ceiling']] = 0
        masks = [mask]+[(gray < t).astype(np.uint8)*255 for t in cfg['thresholds']]
        result = []; height, width = gray.shape
        for candidate_mask in masks:
            contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not cfg['area_min'] <= area <= cfg['area_max']:
                    continue
                _, (w, h), _ = cv2.minAreaRect(contour)
                short, long = sorted((w, h))
                if short < cfg['short_min'] or not cfg['long_min'] <= long <= cfg['long_max']:
                    continue
                if not cfg['ratio_min'] <= long/short <= cfg['ratio_max']:
                    continue
                extent = area/max(w*h, 1.)
                solidity = area/max(cv2.contourArea(cv2.convexHull(contour)), 1.)
                if extent < cfg['extent_min'] or solidity < cfg['solidity_min']:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                px, py = bw*cfg['padding'], bh*cfg['padding']
                box = [max(0., x-px), max(0., y-py), min(float(width), x+bw+px), min(float(height), y+bh+py)]
                result.append({'box': box, 'score': float(extent*solidity)})
        candidates = nms(result, cfg['nms_iou'], cfg['max_candidates'])
        if self.model is None or not candidates:
            return candidates
        filtered = []
        with torch.inference_mode():
            for start in range(0, len(candidates), cfg['inference_batch']):
                batch = candidates[start:start+cfg['inference_batch']]
                patches = np.stack([vehicle_patch(rgb, c['box']) for c in batch])
                tensor = torch.from_numpy(patches.transpose(0, 3, 1, 2).copy()).float()/255.
                scores = self.model(tensor).softmax(1)[:, 1].numpy()
                filtered.extend({'box': c['box'], 'score': float(score)} for c, score in zip(batch, scores)
                                if score >= cfg['confidence'])
        return filtered


def binary_model():
    model = VehicleAppearance()
    model.head[-1] = torch.nn.Linear(64, 2)
    return model


def build_detector(config):
    """汇总分支通过配置字典构建具有 reset/detect 接口的检测器。"""
    if isinstance(config, (str, Path)):
        config = common.read_json(config)
    return StaticDetector(config)


def metrics(manifest, records, split):
    def one(m, fov=None):
        result = common.evaluate(m, records, split=split, fov=fov)
        result['static_gt'] = result['gt']-result['moving_gt']
        result['static_tp'] = result['tp']-result['moving_tp']
        result['static_recall'] = result['static_tp']/result['static_gt'] if result['static_gt'] else None
        # 子集误报沿用全部车辆匹配后的剩余，避免另一子集真阳性变成误报。
        result['subset_false_positive_rule'] = 'unmatched after all-vehicle matching'
        return result
    return {'all': one(manifest), 'fov48': one(manifest, 48), 'fov30': one(manifest, 30),
            'clips': {c['id']: one(dict(manifest, clips=[c])) for c in manifest['clips'] if c['split'] == split}}


def run_stage(manifest, config, output, name, split):
    directory = Path(output)/name
    common.write_json(directory/'config.json', config)
    records = common.run_detector(manifest, lambda: build_detector(config), interval=config['interval'], split=split)
    common.save_records(directory/(split+'_records.jsonl'), records)
    result = metrics(manifest, records, split)
    common.write_json(directory/(split+'_metrics.json'), result)
    print(json.dumps({'stage': name, 'split': split, **{k: v for k, v in result['all'].items() if k != 'event_details'}}, ensure_ascii=False), flush=True)
    return records, result


def training_data(manifest, records, output, seed):
    """仅在 tune 采样帧裁剪车辆、抖动车辆、候选背景和空区域。"""
    rng = np.random.default_rng(seed)
    index = {r['key']: r for r in records}
    patches = []; labels = []; provenance = []; counts = {}
    def add(rgb, frame, box, label, kind):
        patches.append(vehicle_patch(rgb, box)); labels.append(label)
        counts[kind] = counts.get(kind, 0)+1
        provenance.append({'key': frame['key'], 'box': list(map(float, box)), 'label': label, 'kind': kind})
    for clip in manifest['clips']:
        if clip['split'] != 'tune':
            continue
        for frame in common.sample_frames(clip, .3):
            rgb = common.load_rgb(frame); height, width = rgb.shape[:2]
            gt = frame['labels']; candidates = index[frame['key']]['candidates']
            for g in gt:
                b = g['box']; add(rgb, frame, b, 1, 'vehicle_gt')
                bw, bh = b[2]-b[0], b[3]-b[1]; cx, cy = (b[2]+b[0])/2, (b[3]+b[1])/2
                for _ in range(3):
                    dx, dy = rng.uniform(-.15, .15, 2)*[bw, bh]
                    jw, jh = np.array([bw, bh])*rng.uniform(.8, 1.2, 2)
                    jitter = [max(0., cx+dx-jw/2), max(0., cy+dy-jh/2),
                              min(float(width), cx+dx+jw/2), min(float(height), cy+dy+jh/2)]
                    add(rgb, frame, jitter, 1, 'vehicle_jitter')
            # 候选匹配正例补齐实际裁剪分布，每个标签每帧最多一个。
            for i in common.match(candidates, gt):
                add(rgb, frame, candidates[i]['box'], 1, 'vehicle_candidate')
            negatives = [c['box'] for c in candidates if max((common.iou(c['box'], g['box']) for g in gt), default=0.) < .05]
            # 高分轮廓和随机候选各取一半，避免仅学容易的背景。
            chosen = list(range(min(12, len(negatives))))
            if len(negatives) > 12:
                chosen.extend(rng.choice(np.arange(12, len(negatives)), min(12, len(negatives)-12), replace=False).tolist())
            for i in chosen:
                add(rgb, frame, negatives[i], 0, 'candidate_background')
            empty = 0
            for _ in range(40):
                bw, bh = rng.uniform(6., 34., 2); x, y = rng.uniform(0., width-bw), rng.uniform(0., height-bh)
                box = [x, y, x+bw, y+bh]
                if max((common.iou(box, g['box']) for g in gt), default=0.) >= .05:
                    continue
                add(rgb, frame, box, 0, 'empty_background'); empty += 1
                if empty == 4:
                    break
    common.save_records(Path(output)/'training_samples.jsonl', provenance)
    info = {'seed': seed, 'source_split': 'tune', 'source_frames': len(index), 'counts': counts,
            'positive': sum(labels), 'negative': len(labels)-sum(labels), 'total': len(labels),
            'negative_max_iou': .05, 'jitter_per_gt': 3, 'candidate_positive_iou': .25,
            'validation': 'tune is used for fitting and selection; no independent validation or test access'}
    common.write_json(Path(output)/'training_data.json', info)
    return np.stack(patches), np.asarray(labels, np.int64), info


def train_a2(manifest, records, output, proposal_config):
    directory = Path(output)/'A2'; directory.mkdir(parents=True, exist_ok=True)
    config = dict(proposal_config, variant='A2', confidence=.5, inference_batch=128,
                  model_path=str((directory/'model.pt').resolve()))
    recipe = {'seed': 20260913, 'epochs': 12, 'batch_size': 64, 'learning_rate': .001,
              'optimizer': 'Adam', 'weight_decay': .0001, 'loss': 'class-balanced cross entropy',
              'selection': 'fixed final epoch; no threshold sweep; no early stopping', 'training_runs': 1}
    common.write_json(directory/'training_recipe.json', recipe)
    torch.manual_seed(recipe['seed']); np.random.seed(recipe['seed'])
    x, y, info = training_data(manifest, records, directory, recipe['seed'])
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x.transpose(0, 3, 1, 2).copy()), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=recipe['batch_size'], shuffle=True,
                                       num_workers=0, generator=torch.Generator().manual_seed(recipe['seed']))
    model = binary_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    class_counts = np.bincount(y, minlength=2)
    weights = torch.tensor(len(y)/(2*class_counts), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    history = []
    for epoch in range(recipe['epochs']):
        model.train(); total_loss = 0.; correct = 0
        for images, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.float()/255.)
            loss = loss_fn(logits, target); loss.backward(); optimizer.step()
            total_loss += float(loss.detach())*len(target)
            correct += int((logits.argmax(1) == target).sum())
        row = {'epoch': epoch+1, 'loss': total_loss/len(y), 'training_accuracy': correct/len(y)}
        history.append(row)
        common.write_json(directory/'training_history.json', history)
        print(json.dumps({'training': row}, ensure_ascii=False), flush=True)
    torch.save(model.state_dict(), config['model_path'])
    common.write_json(directory/'config.json', config)
    return config, dict(recipe, data=info, history=history,
                        model_sha256=hashlib.sha256(Path(config['model_path']).read_bytes()).hexdigest())


def finish(manifest, manifest_path, output):
    """冻结发生在任何 test 推理之前，已有测试标记时拒绝原目录重跑。"""
    output = Path(output)
    if (output/'test_started.json').exists():
        raise RuntimeError('测试已启动过，请用 replay 模式和新输出目录重跑冻结配置。')
    a1_records = common.read_records(output/'A1_round2/tune_records.jsonl')
    a0_metrics = common.read_json(output/'A0/tune_metrics.json')
    a1_first_metrics = common.read_json(output/'A1_round1/tune_metrics.json')
    a1_metrics = common.read_json(output/'A1_round2/tune_metrics.json')
    a1_config = common.read_json(output/'A1_round2/config.json')
    common.save_records(output/'A1/tune_records.jsonl', a1_records)
    common.write_json(output/'A1/tune_metrics.json', a1_metrics)
    variants = {'A0': {'config': common.read_json(output/'A0/config.json'), 'tune': a0_metrics},
                'A1': {'config': a1_config, 'tune': a1_metrics}}
    training = {'training_runs': 0, 'epochs': 0, 'data': {'total': 0}}
    rounds = [{'round': 0, 'stage': 'A0', 'change': '原始候选绕开分类器', 'metrics': a0_metrics},
              {'round': 1, 'stage': 'A1_round1', 'change': '放宽尺寸比例轮廓并做原图 NMS，最多 512 框', 'metrics': a1_first_metrics},
              {'round': 2, 'stage': 'A1_round2', 'change': '诊断确认截断是移动漏检主因，仅将上限改为 2048', 'metrics': a1_metrics}]
    selected = 'A1'
    enough = (a1_metrics['all']['recall'] >= .8 and a1_metrics['all']['moving_recall'] >= .8
              and a1_metrics['all']['moving_recall']-a1_first_metrics['all']['moving_recall'] >= .2)
    if not enough:
        reason = '第二轮候选未明显改善或召回上限仍不足，停止分类器训练并跳过 A2。'
    else:
        config, training = train_a2(manifest, a1_records, output, a1_config)
        _, result = run_stage(manifest, config, output, 'A2', 'tune')
        variants['A2'] = {'config': config, 'tune': result}
        rounds[-1]['fixed_classifier_validation'] = {'stage': 'A2', 'change': '不再调候选或扫描阈值，一次固定二分类验证', 'metrics': result}
        # 预先固定过滤验收规则：最多损失十个百分点召回且误报至少减半。
        if (all(result['all'][key] >= a1_metrics['all'][key]-.1 for key in ('recall', 'moving_recall'))
                and result['all']['fp_per_frame'] <= a1_metrics['all']['fp_per_frame']*.5):
            selected = 'A2'
            reason = 'A2 tune 两项召回损失均不超过 0.1 且误报至少减半；按过滤验证前固定的规则选择 A2。'
        else:
            reason = 'A2 tune 未同时满足召回损失不超过 0.1 与误报至少减半；保留 A1，不继续优化分类器。'
    interrupted_path = output/'A2_interrupted/training_history.json'
    interrupted = common.read_json(interrupted_path) if interrupted_path.exists() else []
    training['completed_training_runs'] = training['training_runs']
    training['interrupted_attempts'] = int(bool(interrupted))
    training['attempts_total'] = training['training_runs']+int(bool(interrupted))
    training['interrupted_epochs_completed'] = len(interrupted)
    training['interruption_reason'] = '用户要求优先检查候选截断，已中止早先训练，无模型、无 A2 指标、无 test 推理。'
    frozen = {'selected': selected, 'reason': reason, 'config': variants[selected]['config'],
              'variants': {name: row['config'] for name, row in variants.items()},
              'selection_rule': 'A2 all/moving recall loss <= 0.1 and FP/frame <= 0.5*A1; otherwise A1',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
              'parameter_change_rounds': len(rounds)-1, 'training': training}
    common.write_json(output/'tune_rounds.json', rounds)
    common.write_json(output/'frozen_selection.json', frozen)
    common.write_json(output/'config.json', frozen['config'])
    print(json.dumps({'frozen': selected, 'reason': reason}, ensure_ascii=False), flush=True)
    common.write_json(output/'test_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                                                  'passes_per_variant': 1, 'variants': list(variants)})
    final_records = []
    for name, row in variants.items():
        records, result = run_stage(manifest, row['config'], output, name, 'test')
        row['test'] = result
        if name == selected:
            final_records = records
        common.write_videos(manifest, records, output/name/'videos_test', name, interval=.3, split='test')
    common.save_records(output/'final_records.jsonl', final_records)
    summary = dict(frozen, manifest=str(Path(manifest_path).resolve()), interval=.3, cpu_threads=2,
                   detector_input='RGB and source time only; source time ignored', candidate_resolution='original',
                   evaluation='common.run_detector/evaluate; IoU >= 0.25; all vehicle matching first',
                   final_records_split='test', test_runs_per_variant=1, results=variants)
    common.write_json(output/'summary.json', summary)
    write_report(output, summary, rounds)


def write_report(output, summary, rounds):
    def fmt(value):
        return '无样本' if value is None else f'{value:.4f}'
    lines = ['# 静态车辆发现离线实验', '', f"最终冻结方案：{summary['selected']}。{summary['reason']}", '',
             '固定 manifest；每个片段按公共 0.3 秒时间网格采样；检测只接收原分辨率 RGB 与时间（不使用时间）。',
             '真实车辆与诱饵统一为 vehicle，投影框作为正确标注，空框视为背景。IoU 门槛 0.25；静止/移动子集共用全部车辆匹配后的剩余误报。', '',
             '|阶段|划分|帧数|全部召回|移动召回|静止召回|误报/帧|1秒发现率|移动1秒发现率|检测毫秒/帧|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['results'].items():
        for split in ('tune', 'test'):
            m = row[split]['all']
            lines.append(f"|{name}|{split}|{m['frames']}|"+'|'.join(fmt(m[k]) for k in
                         ('recall', 'moving_recall', 'static_recall', 'fp_per_frame', 'discovery_1s', 'moving_discovery_1s', 'mean_ms'))+'|')
    lines += ['', '## 冻结与训练', '', 'A0 是原始 visual_appearance.vehicle_proposals，不加载外观分类器。',
              'A1 保留暗色阈值候选，在原图放宽面积、长短边、比例、矩形填充率与凸包实心度，加边距后按几何分数 NMS；首轮上限 512，最终上限 2048。',
              '第一轮采用 512 候选；排序截断诊断发现移动漏检 14 次中有 10 次由截断导致，原始轮廓和完整 NMS 均命中 61/72 全部框与 31/35 移动框。',
              '未截断 NMS 的每帧候选中位数 1390.5、最大值 3788；匹配车辆最差排名 1628。第二轮仅将上限增至 2048，保持其它参数与排序完全一致。',
              '第二轮明显改善后才进行一次固定 A2 验证；复用 VehicleAppearance 四层卷积，最后输出层改为背景/车辆二分类，不再调整候选与扫描分类阈值。',
              '验证前固定选择规则：A2 相对 A1 的全部/移动召回损失均不超过 0.1 且误报至少减半才采用，否则保留 A1 并停止分类器优化；分类置信度固定 0.5。',
              f"候选参数变化轮数：{summary['parameter_change_rounds']}；完成训练次数：{summary['training']['completed_training_runs']}；完成训练 epoch：{summary['training']['epochs']}。",
              f"另有早先中止训练 {summary['training']['interrupted_attempts']} 次，已完成 {summary['training']['interrupted_epochs_completed']} epoch；收到优先诊断截断的指令后立即中止，未保存模型、未生成 A2 指标、未运行 test，历史保存在 A2_interrupted。总训练启动次数为 {summary['training']['attempts_total']}。",
              '训练数据统计：`'+json.dumps(summary['training']['data'], ensure_ascii=False)+'`。',
              '训练仅使用 tune 的 0.3 秒采样帧；车辆原框与抖动正例、匹配候选正例、候选背景与空区域负例均保存来源。',
              'tune 同时用于训练和方案选择，其指标不是独立泛化验证。test 在 frozen_selection.json/config.json 写入后，每个方案只运行一次；未据此改参数。', '',
              '## 分视场测试', '', '|阶段|FOV|全部召回|移动召回|误报/帧|', '|---|---|---:|---:|---:|']
    for name, row in summary['results'].items():
        for group in ('fov48', 'fov30'):
            m = row['test'][group]
            lines.append(f"|{name}|{group}|{fmt(m['recall'])}|{fmt(m['moving_recall'])}|{fmt(m['fp_per_frame'])}|")
    lines += ['', '## 限制与产物', '',
              '这是固定录制数据上的离线发现实验，不代表新仿真、实机验证或比赛得分。A2 只能删除候选，无法补回 A1 漏掉的车辆。',
              '第一轮 A1 tune 每帧均达到 512 上限，几何得分使岩石、草地和阴影占据前排；第二轮提高上限仅缓解排序截断，仍有原始暗色轮廓缺失，尤其短暂静止车辆片段。分类器无法恢复这些漏检。',
              '公共耗时只统计 detect，包含候选生成与分类，排除读图、每片段模型加载和叠框视频；CPU 算子线程与 OpenCV 线程设为 2。',
              'config.json 和 build_detector(config) 是最终重跑接口；模型采用绝对路径。final_records.jsonl 仅包含冻结方案 test 记录。',
              '各阶段目录保存配置、tune/test records JSONL、指标和公共 write_videos 生成的逐片段 test 视频；绿框为投影标注，橙框为候选。',
              'tune_rounds.json 保存每轮全部/分视场/分片段统计；frozen_selection.json 保存选择理由、冻结时间和 manifest 指纹；summary.json 汇总结果。', '']
    lines += ['## 1 秒发现事件分母', '',
              '公共 evaluate 将每段连续可见区间视为事件，包含片段开始时已在画面内的 clip-entry。可见不足 1 秒的短事件仍在分母内，因此以下不是仅统计新进入车辆的发现率。',
              '多个阶段使用相同事件分母；少量片段与少量事件限制泛化解释。', '', '|划分|全部事件|其中 clip-entry|移动事件|移动 clip-entry|', '|---|---:|---:|---:|---:|']
    for split in ('tune', 'test'):
        events = summary['results']['A1'][split]['all']['event_details']
        lines.append(f"|{split}|{len(events)}|{sum(e['clip_entry'] for e in events)}|{sum(e['moving'] for e in events)}|{sum(e['moving'] and e['clip_entry'] for e in events)}|")
    lines += ['', '|阶段|划分|全部 1 秒命中/事件|移动 1 秒命中/事件|', '|---|---|---:|---:|']
    for name, row in summary['results'].items():
        for split in ('tune', 'test'):
            m = row[split]['all']
            lines.append(f"|{name}|{split}|{m['found_1s']}/{m['events']}|{m['moving_found_1s']}/{m['moving_events']}|")
    lines += ['', '## 各轮 tune 统计', '', '|轮次|全部命中/框|移动命中/框|静止命中/框|误报/帧|', '|---|---:|---:|---:|---:|']
    for row in rounds:
        m = row['metrics']['all']
        lines.append(f"|{row['stage']}|{m['tp']}/{m['gt']}|{m['moving_tp']}/{m['moving_gt']}|{m['static_tp']}/{m['static_gt']}|{fmt(m['fp_per_frame'])}|")
    worktree = Path(__file__).resolve().parents[4]
    interpreter = worktree.parent/'hf2026-sim-windows/ZqhjGame/.venv-learning/Scripts/python.exe'
    lines += ['', '## 冻结方案重跑', '', '在指定独立 worktree 执行，重跑写入新目录，不覆盖本次一次性 test 证据。', '', '```powershell',
              f"Set-Location '{worktree.as_posix()}'",
              f"& '{interpreter.as_posix()}' -B -m competition.user_algorithms.coop_decoy.vehicle_discovery.static `",
              f"  --mode replay --split test --manifest '{summary['manifest']}' `",
              f"  --config '{(Path(output)/'config.json').as_posix()}' `",
              f"  --output '{(Path(output).parent/'static-replay').as_posix()}'", '```', '']
    (Path(output)/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['tune', 'finish', 'replay'], default='tune')
    parser.add_argument('--config')
    parser.add_argument('--split', choices=['tune', 'test'], default='test')
    args = parser.parse_args()
    manifest = common.read_json(args.manifest)
    output = Path(args.output)
    if args.mode == 'tune':
        run_stage(manifest, A0, output, 'A0', 'tune')
        run_stage(manifest, A1, output, 'A1_round1', 'tune')
        run_stage(manifest, A1_ROUND2, output, 'A1_round2', 'tune')
    elif args.mode == 'finish':
        finish(manifest, args.manifest, output)
    else:
        config = common.read_json(args.config)
        records, result = run_stage(manifest, config, output, 'replay', args.split)
        common.save_records(output/'final_records.jsonl', records)
        common.write_json(output/'summary.json', result)
        common.write_videos(manifest, records, output/'videos', config['variant'], interval=config['interval'], split=args.split)


if __name__ == '__main__':
    main()
