#!/usr/bin/env python3
"""Controlled diagnostics of the Omega-512 focal bias; writes only new probe artifacts.

Uses the original 20 full submaps and their cached undistorted baseline.
Controls: same 6% zoom with/without centering principal point, input resolution,
and full-fp32 aggregator. Camera head is unchanged; depth head is skipped only
after loading all checkpoint keys strictly. Baseline reproductions verify this.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vos import env
import numpy as np

OUT = ROOT / 'runs/focal_cause_probe'
RAW = env.TUM_ROOT
UND = ROOT / 'data/tum_undistorted'
SEQS = {s: 'rgbd_dataset_freiburg1_' + s for s in ('desk', 'desk2', 'room')}


def cases_and_jobs():
    calibration_file = ROOT / 'runs/undistort/undistort_report.json'
    if not calibration_file.is_file():
        raise RuntimeError('Run tools/undistort_tum.py and tools/analysis_undistort.py forward first; see docs/REPRODUCE.md')
    calibration = json.loads(calibration_file.read_text())
    k = np.array(calibration['K_new'])
    zoom = k.copy()
    zoom[0, 0] *= 1.06
    zoom[1, 1] *= 1.06
    center = zoom.copy()
    center[0, 2], center[1, 2] = 320., 240.
    cases = {
        'baseline_verify': {'hw': [448, 592], 'K': k.tolist(), 'subset': True},
        'zoom_control': {'hw': [448, 592], 'K': zoom.tolist()},
        'zoom_centered': {'hw': [448, 592], 'K': center.tolist()},
        'resolution_512x384': {'hw': [384, 512], 'K': k.tolist()},
        'resolution_640x480': {'hw': [480, 640], 'K': k.tolist()},
        'fp32': {'hw': [448, 592], 'K': k.tolist(), 'subset': True},
    }
    for scale in (1.25, 1.50):
        kk = k.copy()
        kk[0,0] *= scale
        kk[1,1] *= scale
        kk[0,2], kk[1,2] = 320., 240.
        cases['zoom_centered_'+str(int(scale*100))] = {'hw':[448,592], 'K':kk.tolist(), 'subset':True}
    jobs = []
    for seq in SEQS:
        paths = sorted((ROOT / 'runs/undistort/analysis/cache/omega').glob('A_' + seq + '_*.npz'))
        full = []
        for p in paths:
            with np.load(p) as z:
                if len(z['names']) == 17:
                    full.append({'tag': p.stem, 'seq': seq, 'names': z['names'].tolist()})
        for i, job in enumerate(full):
            job['subset'] = i in (0, len(full) // 2)
            jobs.append(job)
    if len(jobs) != 20:
        raise RuntimeError(f'Expected 20 full baseline windows, found {len(jobs)}; complete tools/analysis_undistort.py forward first')
    return calibration, cases, jobs


def forward():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration, cases, jobs = cases_and_jobs()
    design = {'purpose': 'Exploratory mechanism checks, not a new confirmatory benchmark',
              'cases': cases, 'jobs': jobs,
              'controls': 'zoom_control and zoom_centered have identical focal lengths and resolution; both remapped from raw once; only principal point differs',
              'limitations': 'centering changes the visible boundary; resolution tests keep checkpoint fixed but change aspect ratio by 0.90%; fp32 uses only two windows per sequence'}
    (OUT / 'intervention_design.json').write_text(json.dumps(design, indent=2))
    env.prepare()
    env.claim_gpu(9200, 'Omega focal-cause controlled probe')
    import cv2
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from vos.omega_backend import load_omega, decode_pose
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(9.0 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    model = load_omega()
    model.dense_head = None
    model.eval()
    log_file = OUT / 'intervention_forward.json'
    logs = json.loads(log_file.read_text())['entries'] if log_file.exists() else []
    old_k = np.array(calibration['K_old'])
    distortion = np.array(calibration['D_old_k1_k2_p1_p2_k3'])
    for name, case in cases.items():
        needed = [j for j in jobs if not case.get('subset') or j['subset']]
        if all((OUT/'interventions'/name/(j['tag']+'.npz')).exists() for j in needed):
            continue
        h, w = case['hw']
        maps = None
        if name.startswith('zoom_'):
            maps = cv2.initUndistortRectifyMap(old_k, distortion, None, np.array(case['K']), (640,480), cv2.CV_32FC1)
            inside = (maps[0] >= 0) & (maps[0] <= 639) & (maps[1] >= 0) & (maps[1] <= 479)
            case['source_inside_fraction'] = float(inside.mean())
            print(name, 'inside fraction', case['source_inside_fraction'], flush=True)
        model.autocast = name != 'fp32'
        if name == 'fp32':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        for job in jobs:
            if case.get('subset') and not job['subset']:
                continue
            path = OUT / 'interventions' / name / (job['tag'] + '.npz')
            if path.exists():
                continue
            t0 = time.perf_counter()
            batch = []
            for filename in job['names']:
                root = RAW if maps is not None else UND
                img_path = root / SEQS[job['seq']] / 'rgb' / filename
                with Image.open(img_path) as source:
                    img = source.convert('RGB')
                if maps is not None:
                    img = Image.fromarray(cv2.remap(np.asarray(img), maps[0], maps[1], cv2.INTER_LINEAR))
                img = img.resize((w,h), Image.Resampling.BICUBIC)
                batch.append(to_tensor(img))
            images = torch.stack(batch).cuda()
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.inference_mode():
                    pred = model(images)
                    extrinsic, intrinsic = decode_pose(pred['pose_enc'], (h,w))
                torch.cuda.synchronize()
                payload = {'names': np.array(job['names']), 'pose_enc': pred['pose_enc'].float().cpu().numpy()[0],
                           'extrinsic': extrinsic.float().cpu().numpy()[0], 'intrinsic': intrinsic.float().cpu().numpy()[0],
                           'image_hw': [h,w], 'K_source': case['K']}
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **payload)
                log = {'case': name, 'tag': job['tag'], 'seconds': time.perf_counter()-t0,
                       'peak_gib': torch.cuda.max_memory_allocated()/1024**3, 'status': 'ok'}
            except torch.cuda.OutOfMemoryError as e:
                log = {'case': name, 'tag': job['tag'], 'status': 'oom', 'error': str(e)}
            finally:
                images = pred = extrinsic = intrinsic = None
                torch.cuda.empty_cache()
            logs.append(log)
            (OUT / 'intervention_forward.json').write_text(json.dumps({'cases':cases, 'entries': logs}, indent=2))
            print(json.dumps(log), flush=True)
    print('FORWARD COMPLETE', flush=True)


def analyze():
    from vos import why_utils as wu
    from scipy.spatial.transform import Rotation
    calibration, cases, jobs = cases_and_jobs()
    gt = {s: wu.load_groundtruth(RAW / directory / 'groundtruth.txt') for s,directory in SEQS.items()}
    cases = {'omega_cached': {'hw':[448,592], 'K': calibration['K_new']},
             'vggt_cached': {'hw':[392,518], 'K': calibration['K_new']}, **cases}
    rows = []
    baseline_diff = []
    for case, info in cases.items():
        for job in jobs:
            if info.get('subset') and not job['subset']:
                continue
            if case.endswith('_cached'):
                path = ROOT / 'runs/undistort/analysis/cache' / case.split('_')[0] / (job['tag']+'.npz')
            else:
                path = OUT / 'interventions' / case / (job['tag']+'.npz')
            if not path.exists():
                continue
            with np.load(path) as z:
                pe = z['pose_enc'].astype(float)
                # Re-decode quaternion and FoV in fp64 for both backends.
                e = np.concatenate([Rotation.from_quat(pe[:,3:7]).as_matrix(), pe[:,:3,None]],axis=2)
                h,w = z['image_hw']
                k = z['intrinsic'].astype(float)
                k[:,0,0] = w/2 / np.tan(pe[:,8]/2)
                k[:,1,1] = h/2 / np.tan(pe[:,7]/2)
                metrics = wu.trajectory_metrics(e,k,(h,w),[wu.stamp_of(n) for n in z['names']],gt[job['seq']])
                true_k = np.array(info['K'])
                row = {'case':case,'tag':job['tag'],'seq':job['seq'],
                       'fx_bias': float(k[:,0,0].mean()/w/(true_k[0,0]/640)-1),
                       'fy_bias': float(k[:,1,1].mean()/h/(true_k[1,1]/480)-1),
                       'ate':metrics['ate_rmse_m'], 'ate_path':metrics['ate_over_gt_path'],
                       'rel_rot_deg':metrics['rel_rot_err_deg_mean'], 'rel_dir_deg':metrics['rel_dir_err_deg_mean']}
                rows.append(row)
                if case == 'baseline_verify':
                    with np.load(ROOT/'runs/undistort/analysis/cache/omega'/(job['tag']+'.npz')) as old:
                        baseline_diff.append({'tag':job['tag'],'max_pose_enc_abs_diff':float(np.max(np.abs(z['pose_enc']-old['pose_enc'])))})
    summary = {}
    ref = {r['tag']:r for r in rows if r['case']=='omega_cached'}
    for case in cases:
        rr = [r for r in rows if r['case']==case]
        if not rr:
            continue
        summary[case] = {'n':len(rr), 'fx_bias_mean':float(np.mean([r['fx_bias'] for r in rr])),
                         'fy_bias_mean':float(np.mean([r['fy_bias'] for r in rr])),
                         'ate_path_median':float(np.median([r['ate_path'] for r in rr])),
                         'paired_ate_ratio_to_cached_omega_median':float(np.median([r['ate']/ref[r['tag']]['ate'] for r in rr])),
                         'paired_fx_change_pp_mean':float(np.mean([100*(r['fx_bias']-ref[r['tag']]['fx_bias']) for r in rr]))}
        summary[case]['per_sequence'] = {s:{'fx_bias_mean':float(np.mean([r['fx_bias'] for r in rr if r['seq']==s])),
                                           'ate_ratio_median':float(np.median([r['ate']/ref[r['tag']]['ate'] for r in rr if r['seq']==s]))} for s in SEQS}
    by_case = {c:{r['tag']:r for r in rows if r['case']==c} for c in cases}
    pairs = {}
    for treatment, control in [('zoom_centered','zoom_control'), ('fp32','omega_cached'),
                               ('zoom_centered_125','zoom_centered'), ('zoom_centered_150','zoom_centered')]:
        tags = sorted(set(by_case[treatment]) & set(by_case[control]))
        if tags:
            pairs[treatment+'_vs_'+control] = {
                'n':len(tags),
                'mean_fx_bias_difference_pp':float(np.mean([100*(by_case[treatment][t]['fx_bias']-by_case[control][t]['fx_bias']) for t in tags])),
                'median_ate_ratio':float(np.median([by_case[treatment][t]['ate']/by_case[control][t]['ate'] for t in tags])),
                'control_fx_bias_mean':float(np.mean([by_case[control][t]['fx_bias'] for t in tags])),
                'treatment_fx_bias_mean':float(np.mean([by_case[treatment][t]['fx_bias'] for t in tags])),
                'n_ate_improves':sum(by_case[treatment][t]['ate'] < by_case[control][t]['ate'] for t in tags)}
    out = {'baseline_verification':baseline_diff,'summary':summary,'direct_paired_controls':pairs,'rows':rows,
           'metric_note':'Re-decoded cached quaternion and FoV in fp64; ATE uses Sim3; exploratory windows overlap, not independent samples.'}
    (OUT/'intervention_results.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'baseline_verification':baseline_diff,'summary':summary,'direct_paired_controls':pairs},indent=2))
    assert all(v['max_pose_enc_abs_diff']==0 for v in baseline_diff), 'Baseline pose differs from original forward'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['forward','analyze'])
    args = parser.parse_args()
    if args.stage == 'forward':
        forward()
    else:
        analyze()
