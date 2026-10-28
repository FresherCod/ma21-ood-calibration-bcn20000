# -*- coding: utf-8 -*-
"""
Đối chứng màu tách khỏi độ sáng và độ dốc (thay thế nhánh khớp histogram RGB của imagelevel_arms.py).

Các nhánh (mỗi nhánh là ảnh ID test đã biến đổi, chấm bằng năng lượng và MSP sau nhiệt độ T):
  orig        ID gốc
  chroma      khớp trung bình và độ lệch chuẩn của kênh a, b (không gian Lab) với bộ far; giữ nguyên L.
              Bộ far xám có a = b = 0 nên nhánh này khử màu.
  chroma_rot  placebo màu: như chroma nhưng đích là chroma của bộ far đã XOAY một góc ngẫu nhiên trong
              (60°, 300°). Cùng độ lớn chroma, hướng không hướng về bộ far. Chỉ áp dụng cho bộ far có màu.
  lum         khớp trung bình và độ lệch chuẩn của L (độ sáng, tương phản) với bộ far; giữ a, b.
  full        chroma rồi biến đổi độ dốc (làm mờ hoặc làm sắc nét) để khớp median độ dốc của bộ far.
  placebo_t   ID gốc chỉ áp dụng cùng biến đổi độ dốc của full (không khớp gì).

Đọc kết quả (quy tắc đặt trước; xem verdict()):
  - Tín hiệu gốc phải khác 0,5 (CI không chứa 0,5), nếu không thì không có gì để giải thích.
  - Nhánh chroma: giải thích được bởi màu nếu AUROC khớp nằm trong 0,5 ± 0,05 (CI nằm trọn trong đó),
    shift nằm trong 0,5 ± 0,05 và |SMD độ bão hòa| <= 0,2.
  - Đặc hiệu theo hướng: chroma - chroma_rot (cùng độ lớn). CI không chứa 0 nghĩa là hướng về bộ far có ý nghĩa.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from skimage.color import lab2rgb, rgb2lab

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_calibration as m  # noqa: E402
import imagelevel_control as ic  # noqa: E402
import imagelevel_arms as ia  # noqa: E402  (apply_transform, logit_scores, auroc_pair)

OUT = os.environ.get('OUT', os.path.join(m.RUN_ROOT, 'imagelevel_chroma'))
ARMS = ['orig', 'chroma', 'chroma_rot', 'lum', 'full', 'placebo_t']


def to_lab(imgs):
    out = np.empty(imgs.shape, np.float32)
    for i in range(0, len(imgs), 25):
        out[i:i + 25] = rgb2lab(imgs[i:i + 25].astype(np.float64) / 255.0).astype(np.float32)
    return out


def from_lab(lab):
    out = np.empty(lab.shape, np.uint8)
    for i in range(0, len(lab), 25):
        rgb = lab2rgb(lab[i:i + 25].astype(np.float64))
        out[i:i + 25] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return out


def lab_stats(lab):
    return {f'{name}_{stat}': float(getattr(np, stat)(lab[..., c]))
            for name, c in [('L', 0), ('a', 1), ('b', 2)] for stat in ['mean', 'std']}


def rotate_chroma(lab, theta):
    out = lab.copy()
    a, b = lab[..., 1], lab[..., 2]
    out[..., 1] = a * np.cos(theta) - b * np.sin(theta)
    out[..., 2] = a * np.sin(theta) + b * np.cos(theta)
    return out


def _match(x, mu_t, sd_t):
    """Khớp trung bình và độ lệch chuẩn của x với (mu_t, sd_t). Nếu đích hằng số thì gán bằng mu_t."""
    if sd_t < 1e-3:
        return np.full_like(x, mu_t)
    mu, sd = float(x.mean()), float(x.std())
    if sd < 1e-6:
        return np.full_like(x, mu_t)
    return (x - mu) * (sd_t / sd) + mu_t


def chroma_match(lab, st):
    out = lab.copy()
    out[..., 1] = _match(lab[..., 1], st['a_mean'], st['a_std'])
    out[..., 2] = _match(lab[..., 2], st['b_mean'], st['b_std'])
    return out


def lum_match(lab, st):
    out = lab.copy()
    out[..., 0] = np.clip(_match(lab[..., 0], st['L_mean'], st['L_std']), 0.0, 100.0)
    return out


def mean_chroma(lab):
    return float(np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2).mean())


def run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp):
    t0 = time.time()
    S = m.build_split(seed, rowmap)
    K = len(S['classes'])
    y, row_id = S['y'], S['idd']['row'].values
    rng = np.random.default_rng(seed + 100)            # cùng thứ tự RNG với imagelevel_arms (bỏ qua nhánh placebo màu cũ)

    with open(os.path.join(ic.RESULTS_DIR, f'results_seed{seed}.json'), encoding='utf-8') as f:
        T = float(json.load(f)['temperature'])
    model = m.build_model(K).to(device)
    ck = torch.load(os.path.join(ic.CKPT_DIR, f'seed{seed}', 'best_model.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state'])
    model.eval()

    te = S['te']
    te_sub = rng.choice(te, min(ic.N_ID_SUB, len(te)), replace=False)
    id_imgs = np.asarray(X[row_id[te_sub]])
    n = len(id_imgs)
    _, L_id, _ = m.get_feats_logits(model, id_imgs, np.arange(n), np.full(n, -1), device, use_amp)
    s_id = ia.logit_scores(L_id, T)
    lab_id = to_lab(id_imgs)
    sat_id, grad_id = m.lowlevel_scores(id_imgs, np.arange(n))

    out = {'seed': seed, 'T': T, 'n_id': n, 'sources': {}}
    for si, src in enumerate(np.unique(far_src)):
        mk = np.flatnonzero(far_src == src)
        pick = rng.choice(mk, min(ic.N_FAR_SUB, len(mk)), replace=False)
        far_imgs = np.asarray(Xfar[pick])
        nf = len(far_imgs)
        _, L_far, _ = m.get_feats_logits(model, far_imgs, np.arange(nf), np.full(nf, -1), device, use_amp)
        s_far = ia.logit_scores(L_far, T)
        sat_far, grad_far = m.lowlevel_scores(far_imgs, np.arange(nf))
        lab_far = to_lab(far_imgs)
        gray = ic.is_gray_set(far_imgs)
        st = lab_stats(lab_far)

        arms = {'orig': id_imgs}
        chroma_lab = chroma_match(lab_id, st)
        arms['chroma'] = from_lab(chroma_lab)
        arms['lum'] = from_lab(lum_match(lab_id, st))
        arm_chroma_mag = {'chroma': mean_chroma(chroma_lab), 'far': mean_chroma(lab_far)}
        if not gray:
            theta = np.random.default_rng(seed + 300 + si).uniform(np.deg2rad(60), np.deg2rad(300))
            rot_stats = lab_stats(rotate_chroma(lab_far, theta))
            rot_lab = chroma_match(lab_id, rot_stats)
            arms['chroma_rot'] = from_lab(rot_lab)
            arm_chroma_mag['chroma_rot'] = mean_chroma(rot_lab)
            theta_deg = float(np.rad2deg(theta))
        else:
            theta_deg = None
        full, kind, param, g_full = ic.transform_to_grad(arms['chroma'], float(np.median(grad_far)))
        arms['full'] = full
        arms['placebo_t'] = ia.apply_transform(id_imgs, kind, param)

        rec = {'n_far': nf, 'gray_set': gray, 'theta_deg': theta_deg, 'transform': {'kind': kind, 'param': param},
               'grad_target': float(np.median(grad_far)), 'grad_full': g_full,
               'chroma_mean': arm_chroma_mag, 'arms': {}}
        for arm in ARMS:
            if arm == 'chroma_rot' and gray:
                continue
            if arm == 'orig':
                s_arm = s_id
            else:
                _, L_arm, _ = m.get_feats_logits(model, arms[arm], np.arange(n), np.full(n, -1), device, use_amp)
                s_arm = ia.logit_scores(L_arm, T)
            sat_a, grad_a = m.lowlevel_scores(arms[arm], np.arange(n))
            rec['arms'][arm] = {
                'balance_sat_smd': ic.smd(sat_a, sat_far),
                'balance_grad_smd': ic.smd(grad_a, grad_far),
                'auc': {k: {'vs_far': ia.auroc_pair(s_arm[k], s_far[k]),
                            'shift': ia.auroc_pair(s_id[k], s_arm[k])} for k in ['energy_ts', 'msp_ts']},
            }
        out['sources'][str(src)] = rec
        parts = [f'{arm} {v["auc"]["energy_ts"]["vs_far"]:.3f} (shift {v["auc"]["energy_ts"]["shift"]:.3f})'
                 for arm, v in rec['arms'].items()]
        print(f'  seed {seed} {src} ({"xám" if gray else "màu"}): ' + ' | '.join(parts) +
              f' | {kind}({param})', flush=True)
    out['seconds'] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'chroma_seed{seed}.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'seed {seed} xong trong {out["seconds"]} s', flush=True)


def t_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    mu = float(x.mean())
    if n < 2:
        return mu, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return mu, mu - h, mu + h


def load(out_dir):
    rows = []
    for f in sorted(os.listdir(out_dir)):
        if not f.startswith('chroma_seed'):
            continue
        with open(os.path.join(out_dir, f), encoding='utf-8') as fh:
            r = json.load(fh)
        for src, d in r['sources'].items():
            for arm, a in d['arms'].items():
                for k in ['energy_ts', 'msp_ts']:
                    rows.append(dict(seed=r['seed'], source=src, arm=arm, method=k,
                                     vs_far=a['auc'][k]['vs_far'], shift=a['auc'][k]['shift'],
                                     sat_smd=a['balance_sat_smd'], grad_smd=a['balance_grad_smd'],
                                     gray_set=d['gray_set'], transform=f"{d['transform']['kind']}({d['transform']['param']})",
                                     theta_deg=d['theta_deg']))
    return pd.DataFrame(rows)


def verdict(n, orig_ci, arm_ci, arm_shift, sat_smd):
    if n < 2 or np.isnan(arm_ci[0]):
        return 'không kết luận: cần ít nhất 2 seed'
    if orig_ci[0] <= 0.5 <= orig_ci[1]:
        return 'không có tín hiệu gốc để giải thích'
    if abs(arm_shift - 0.5) > 0.05:
        return 'không kết luận: biến đổi tự đổi điểm (shift)'
    if sat_smd is None or np.isnan(sat_smd) or abs(sat_smd) > 0.2:
        return 'không kết luận: cân bằng độ bão hòa kém'
    if 0.45 <= arm_ci[0] and arm_ci[1] <= 0.55:
        return 'giải thích được bởi biến đổi này (CI nằm trong 0,5 ± 0,05)'
    if arm_ci[0] > 0.55:
        return 'tín hiệu còn lại sau biến đổi'
    return 'không kết luận: CI quá rộng'


def summarize(dirs):
    frames = []
    for d in dirs:
        label, path = d.split('=', 1) if '=' in d else (os.path.basename(os.path.normpath(d)), d)
        df = load(path)
        if not df.empty:
            df.insert(0, 'run', label)
            frames.append(df)
    if not frames:
        raise SystemExit('Không có chroma_seed*.json.')
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['run', 'seed', 'source', 'arm', 'method'])
    all_df.to_csv(os.path.join(OUT, 'chroma_by_seed.csv'), index=False)
    rows = []
    for (run, src, arm, meth), g in all_df.groupby(['run', 'source', 'arm', 'method']):
        nseed = int(g['seed'].nunique())
        v = t_ci(g['vs_far']); sh = t_ci(g['shift'])
        ss = t_ci(g['sat_smd'])[0]; sg = t_ci(g['grad_smd'])[0]
        orig = all_df[(all_df.run == run) & (all_df.source == src) & (all_df.arm == 'orig') & (all_df.method == meth)]
        oci = t_ci(orig['vs_far'])
        rows.append(dict(run=run, source=src, arm=arm, method=meth, n=nseed, transform=g['transform'].iloc[0],
                         vs_far=v[0], vs_far_ci=f'[{v[1]:.3f}; {v[2]:.3f}]', shift=sh[0],
                         sat_smd=ss, grad_smd=sg,
                         verdict=verdict(nseed, (oci[1], oci[2]), (v[1], v[2]), sh[0], ss)
                         if (arm in ('chroma', 'lum') and meth == 'energy_ts') else ''))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, 'chroma_summary.csv'), index=False)

    # đặc hiệu theo hướng: chroma - chroma_rot, ghép theo seed
    contr = []
    for (run, src), g in all_df[all_df.method == 'energy_ts'].groupby(['run', 'source']):
        a = g[g.arm == 'chroma'].set_index('seed')['vs_far']
        b = g[g.arm == 'chroma_rot'].set_index('seed')['vs_far']
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        d = (a.loc[common] - b.loc[common]).values
        mu, lo, hi = t_ci(d)
        p = float(stats.ttest_1samp(d, 0.0).pvalue)
        contr.append(dict(run=run, source=src, n=len(common), chroma_minus_rot=mu,
                          ci=f'[{lo:.3f}; {hi:.3f}]', p=p))
    contr_df = pd.DataFrame(contr)
    contr_df.to_csv(os.path.join(OUT, 'chroma_vs_rotated.csv'), index=False)
    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', 30)
    print(summ.round(3).to_string(index=False))
    print()
    print(contr_df.round(3).to_string(index=False))


def main():
    if os.environ.get('SUMMARIZE_DIRS'):
        global OUT
        OUT = os.environ.get('SUMMARIZE_OUT', OUT)
        os.makedirs(OUT, exist_ok=True)
        summarize(os.environ['SUMMARIZE_DIRS'].split(';'))
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    df = pd.read_csv(m.DATA_CSV)
    df = df[df['split'].astype(str).str.lower() == 'train'].reset_index(drop=True)
    names = df['bcn_filename'].tolist()
    rowmap = {f: i for i, f in enumerate(names)}
    X = np.memmap(ic.IMG_MEMMAP, dtype=np.uint8, mode='r', shape=(len(names), m.RESIZE, m.RESIZE, 3))
    Xfar, far_src = m.build_far_array()
    if Xfar is None:
        raise SystemExit('Cần FAR_NPZ để có bộ far-OOD.')
    print('device:', device, '| BACKBONE:', m.BACKBONE, '| nhánh:', ARMS, flush=True)
    for seed in [int(s) for s in m.SEEDS]:
        run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp)


if __name__ == '__main__':
    main()
