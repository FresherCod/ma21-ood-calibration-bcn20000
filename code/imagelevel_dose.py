# -*- coding: utf-8 -*-
"""
Đường đáp ứng liều: đưa sắc độ (a, b trong Lab) của ảnh ID về phía bộ far với liều α, so với placebo vuông góc.
Thiết kế, cổng và quy tắc quyết định: docs/dose_response_design.md (v0.2, đặt trước khi chạy).

Với mỗi bộ far và seed:
  orig                       ảnh ID gốc (liều 0)
  dir_α                      đích trung bình μ_ID + α·d, d = μ_far − μ_ID; đích σ = σ_ID + α(σ_far − σ_ID)
  rot_α (chỉ bộ có màu)      đích trung bình μ_ID + α·R90·d, cùng σ đích với dir_α; R90 (a,b) -> (−b, a)
Bộ xám chỉ có dir_α. Giữ nguyên L ở mọi nhánh.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_calibration as m  # noqa: E402
import imagelevel_control as ic  # noqa: E402
import imagelevel_arms as ia  # noqa: E402
import imagelevel_chroma as ch  # noqa: E402

OUT = os.environ.get('OUT', os.path.join(m.RUN_ROOT, 'imagelevel_dose'))
DOSES = [0.25, 0.5, 1.0]
PRIMARY_ALPHA = '1'
DELTA = 0.05          # biên tương đương
DE_GATE = (0.9, 1.1)  # cổng ΔE76 theo từng seed
BAL_MAX = 0.2         # cổng cân bằng SMD (trung bình 5 seed)
COLOUR_SETS = ('retinamnist_224', 'pathmnist_224', 'bloodmnist_224')
KEYS = ['a_mean', 'a_std', 'b_mean', 'b_std', 'L_mean', 'L_std']


def dir_targets(st_id, st_far, alpha):
    return {k: st_id[k] + alpha * (st_far[k] - st_id[k]) for k in KEYS}


COS_MAX = 0.3          # ràng buộc hình học: |cos(R_θ d, d)| <= 0,3
MAG_TOL = 0.02         # ràng buộc độ lớn chroma: |mc_rot − mc_dir| / mc_dir <= 2%
N_THETA_PIX = 20000    # số pixel lấy mẫu để tìm θ (chỉ dùng thống kê ảnh)


def _rotated_displacement(st_id, st_far, theta_deg):
    d_a = st_far['a_mean'] - st_id['a_mean']
    d_b = st_far['b_mean'] - st_id['b_mean']
    th = np.deg2rad(theta_deg)
    return d_a * np.cos(th) - d_b * np.sin(th), d_a * np.sin(th) + d_b * np.cos(th)


def rot_targets(st_id, st_far, alpha, theta_deg):
    """Đích placebo (v0.3): xoay độ dời trung bình d của far một góc θ; σ đích giống dir_targets."""
    ra, rb = _rotated_displacement(st_id, st_far, theta_deg)
    t = dir_targets(st_id, st_far, alpha)
    t['a_mean'] = st_id['a_mean'] + alpha * ra
    t['b_mean'] = st_id['b_mean'] + alpha * rb
    return t


def _mean_chroma_affine(a, b, mu_a, sd_a, mu_b, sd_b, t):
    a2 = (a - mu_a) * (t['a_std'] / sd_a) + t['a_mean'] if sd_a > 1e-6 else np.full_like(a, t['a_mean'])
    b2 = (b - mu_b) * (t['b_std'] / sd_b) + t['b_mean'] if sd_b > 1e-6 else np.full_like(b, t['b_mean'])
    return float(np.sqrt(a2 ** 2 + b2 ** 2).mean())


def placebo_theta(lab_id, st_id, st_far, alpha, rng):
    """Chọn θ (1° một bước) sao cho |cos| <= COS_MAX và độ lớn chroma trung bình của placebo gần dir nhất.
    Chỉ dùng thống kê ảnh của tập ID, không dùng kết quả mô hình. Trả về None nếu không có θ hợp lệ."""
    d_a = st_far['a_mean'] - st_id['a_mean']
    d_b = st_far['b_mean'] - st_id['b_mean']
    d_norm = float(np.hypot(d_a, d_b))
    if d_norm < 1e-6:
        return None
    a_full, b_full = lab_id[..., 1], lab_id[..., 2]
    mu_a, sd_a = float(a_full.mean()), float(a_full.std())
    mu_b, sd_b = float(b_full.mean()), float(b_full.std())
    fa, fb = a_full.ravel(), b_full.ravel()
    idx = rng.choice(fa.size, min(N_THETA_PIX, fa.size), replace=False)
    a, b = fa[idx], fb[idx]
    dir_t = dir_targets(st_id, st_far, alpha)
    mc_dir = _mean_chroma_affine(a, b, mu_a, sd_a, mu_b, sd_b, dir_t)
    best = None
    for deg in range(360):
        ra, rb = _rotated_displacement(st_id, st_far, deg)
        cos = (ra * d_a + rb * d_b) / (np.hypot(ra, rb) * d_norm + 1e-12)
        if abs(cos) > COS_MAX:
            continue
        t = dict(dir_t)
        t['a_mean'] = st_id['a_mean'] + alpha * ra
        t['b_mean'] = st_id['b_mean'] + alpha * rb
        mc_rot = _mean_chroma_affine(a, b, mu_a, sd_a, mu_b, sd_b, t)
        rel = abs(mc_rot - mc_dir) / max(mc_dir, 1e-9)
        if best is None or rel < best['mag_rel']:
            best = dict(theta_deg=deg, cos=float(cos), mag_rel=float(rel), mc_dir=mc_dir, mc_rot=mc_rot)
    if best is None or best['mag_rel'] > MAG_TOL:
        return {'feasible': False, **(best or {})}
    return {'feasible': True, **best}


def lum_of(imgs):
    f = imgs.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def delta_e76(lab_a, lab_b):
    return float(np.sqrt(((lab_a - lab_b) ** 2).sum(-1)).mean())


def arm_record(imgs, lab_arm, lab_id, s_arm, s_id, s_far, far_refs, n):
    sat_a, grad_a = m.lowlevel_scores(imgs, np.arange(n))
    lum_a = lum_of(imgs)
    rec = {
        'balance_sat_smd': ic.smd(sat_a, far_refs['sat']),
        'balance_lum_smd': ic.smd(lum_a, far_refs['lum']),
        'balance_grad_smd': ic.smd(grad_a, far_refs['grad']),
        'delta_e76': delta_e76(lab_arm, lab_id),
        'chroma_mean': ch.mean_chroma(lab_arm),
        'auc': {},
    }
    for k in ['energy_ts', 'msp_ts']:
        rec['auc'][k] = {'vs_far': ia.auroc_pair(s_arm[k], s_far[k]), 'shift': ia.auroc_pair(s_id[k], s_arm[k])}
    return rec


def run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp):
    t0 = time.time()
    S = m.build_split(seed, rowmap)
    K = len(S['classes'])
    y, row_id = S['y'], S['idd']['row'].values
    rng = np.random.default_rng(seed + 100)

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

    def score(imgs):
        _, L, _ = m.get_feats_logits(model, imgs, np.arange(len(imgs)), np.full(len(imgs), -1), device, use_amp)
        return ia.logit_scores(L, T)

    s_id = score(id_imgs)
    lab_id = ch.to_lab(id_imgs)
    st_id = ch.lab_stats(lab_id)
    out = {'seed': seed, 'T': T, 'n_id': n, 'doses': DOSES, 'sources': {}}

    for src in np.unique(far_src):
        mk = np.flatnonzero(far_src == src)
        pick = rng.choice(mk, min(ic.N_FAR_SUB, len(mk)), replace=False)
        far_imgs = np.asarray(Xfar[pick])
        nf = len(far_imgs)
        s_far = score(far_imgs)
        lab_far = ch.to_lab(far_imgs)
        st_far = ch.lab_stats(lab_far)
        gray = ic.is_gray_set(far_imgs)
        sat_f, grad_f = m.lowlevel_scores(far_imgs, np.arange(nf))
        far_refs = {'sat': sat_f, 'grad': grad_f, 'lum': lum_of(far_imgs)}
        rec = {'n_far': nf, 'gray_set': gray, 'grad_target': float(np.median(grad_f)), 'arms': {}}
        rec['arms']['orig'] = arm_record(id_imgs, lab_id, lab_id, s_id, s_id, s_far, far_refs, n)
        for alpha in DOSES:
            tag = f'{alpha:g}'
            lab_dir = ch.chroma_match(lab_id, dir_targets(st_id, st_far, alpha))
            dir_imgs = ch.from_lab(lab_dir)
            rec['arms'][f'dir_{tag}'] = arm_record(dir_imgs, lab_dir, lab_id, score(dir_imgs), s_id, s_far,
                                                   far_refs, n)
            if not gray:
                pb = placebo_theta(lab_id, st_id, st_far, alpha, rng)
                rec.setdefault('placebo', {})[tag] = pb
                if pb is not None and pb['feasible']:
                    lab_rot = ch.chroma_match(lab_id, rot_targets(st_id, st_far, alpha, pb['theta_deg']))
                    rot_imgs = ch.from_lab(lab_rot)
                    rec['arms'][f'rot_{tag}'] = arm_record(rot_imgs, lab_rot, lab_id, score(rot_imgs), s_id, s_far,
                                                           far_refs, n)
        out['sources'][str(src)] = rec
        print(f'  seed {seed} {src} ({"xám" if gray else "màu"}): ' +
              ' | '.join(f'{k} {v["auc"]["energy_ts"]["vs_far"]:.3f}' for k, v in rec['arms'].items()), flush=True)

    out['seconds'] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'dose_seed{seed}.json'), 'w', encoding='utf-8') as f:
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


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m_ = len(p)
    adj = np.empty(m_)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m_ - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def load(out_dir):
    rows = []
    for f in sorted(os.listdir(out_dir)):
        if not f.startswith('dose_seed'):
            continue
        with open(os.path.join(out_dir, f), encoding='utf-8') as fh:
            r = json.load(fh)
        for src, d in r['sources'].items():
            for arm, a in d['arms'].items():
                for k in ['energy_ts', 'msp_ts']:
                    rows.append(dict(seed=r['seed'], source=src, arm=arm, method=k, gray_set=d['gray_set'],
                                     vs_far=a['auc'][k]['vs_far'], shift=a['auc'][k]['shift'],
                                     sat_smd=a['balance_sat_smd'], lum_smd=a['balance_lum_smd'],
                                     grad_smd=a['balance_grad_smd'], delta_e76=a['delta_e76'],
                                     chroma_mean=a['chroma_mean']))
    return pd.DataFrame(rows)


def summarize(dirs):
    frames = []
    for d in dirs:
        label, path = d.split('=', 1) if '=' in d else (os.path.basename(os.path.normpath(d)), d)
        df = load(path)
        if not df.empty:
            df.insert(0, 'run', label)
            frames.append(df)
    if not frames:
        raise SystemExit('Không có dose_seed*.json.')
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['run', 'seed', 'source', 'arm', 'method'])
    all_df.to_csv(os.path.join(OUT, 'dose_by_seed.csv'), index=False)

    e = all_df[all_df.method == 'energy_ts']
    rows = []
    for (run, src, arm), g in e.groupby(['run', 'source', 'arm']):
        v = t_ci(g['vs_far']); sh = t_ci(g['shift'])
        rows.append(dict(run=run, source=src, arm=arm, n=int(g['seed'].nunique()), gray=bool(g['gray_set'].iloc[0]),
                         vs_far=v[0], vs_far_ci=f'[{v[1]:.3f}; {v[2]:.3f}]', shift=sh[0],
                         sat_smd=t_ci(g['sat_smd'])[0], lum_smd=t_ci(g['lum_smd'])[0],
                         grad_smd=t_ci(g['grad_smd'])[0], delta_e76=g['delta_e76'].mean(),
                         chroma_mean=g['chroma_mean'].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'dose_summary.csv'), index=False)

    # D = dir − rot theo cặp seed; liều α = 1 là phân tích chính, các liều khác là phụ
    dr = []
    for (run, src), g in e.groupby(['run', 'source']):
        if g['gray_set'].iloc[0]:
            continue
        orig = g[g.arm == 'orig'].set_index('seed')['vs_far']
        oci = t_ci(orig.values)
        for alpha in ['0.25', '0.5', '1']:
            dg = g[g.arm == f'dir_{alpha}'].set_index('seed')
            rg = g[g.arm == f'rot_{alpha}'].set_index('seed')
            common = dg.index.intersection(rg.index)
            if len(common) < 2:
                continue
            d = (dg.loc[common, 'vs_far'] - rg.loc[common, 'vs_far']).values
            ratio = (rg.loc[common, 'delta_e76'] / dg.loc[common, 'delta_e76']).values
            gate_de = bool(np.all((ratio >= DE_GATE[0]) & (ratio <= DE_GATE[1])))
            bal = (abs(t_ci(dg.loc[common, 'sat_smd'])[0]) <= BAL_MAX and abs(t_ci(dg.loc[common, 'lum_smd'])[0]) <= BAL_MAX
                   and abs(t_ci(rg.loc[common, 'sat_smd'])[0]) <= BAL_MAX and abs(t_ci(rg.loc[common, 'lum_smd'])[0]) <= BAL_MAX)
            mu, lo, hi = t_ci(d)
            p = float(stats.ttest_1samp(d, 0.0).pvalue)
            dr.append(dict(run=run, source=src, dose=alpha, n=len(common), D_dir_minus_rot=mu,
                           D_ci_lo=lo, D_ci_hi=hi, p=p, orig_signal=not (oci[1] <= 0.5 <= oci[2]),
                           gate_delta_e=gate_de, gate_balance=bal, de_ratio_mean=float(ratio.mean()),
                           primary=(alpha == PRIMARY_ALPHA)))
    dr_df = pd.DataFrame(dr)
    if not dr_df.empty:
        prim = dr_df.index[dr_df.primary & dr_df.source.isin(COLOUR_SETS)]
        dr_df.loc[prim, 'p_holm'] = holm(dr_df.loc[prim, 'p'].values)

        def decide(r):
            if not r['primary']:
                return ''
            if not r['orig_signal']:
                return 'không có tín hiệu gốc'
            if not (r['gate_delta_e'] and r['gate_balance']):
                return 'không kết luận: không đạt cổng ΔE hoặc cân bằng'
            lo, hi = r['D_ci_lo'], r['D_ci_hi']
            if lo >= -DELTA and hi <= DELTA:
                return 'không đặc hiệu theo hướng (CI trong ±δ)'
            if hi < -DELTA:
                return 'đặc hiệu theo hướng về far (dir giảm nhiều hơn)'
            if lo > DELTA:
                return 'đặc hiệu ngược (rot giảm nhiều hơn)'
            return 'không kết luận: CI quá rộng'
        dr_df['decision'] = dr_df.apply(decide, axis=1)
    dr_df.to_csv(os.path.join(OUT, 'dose_decision.csv'), index=False)
    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', 30)
    print(pd.read_csv(os.path.join(OUT, 'dose_summary.csv')).round(3).to_string(index=False))
    print()
    print(dr_df.round(3).to_string(index=False))


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
    print('device:', device, '| BACKBONE:', m.BACKBONE, '| liều:', DOSES, flush=True)
    for seed in [int(s) for s in m.SEEDS]:
        run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp)


if __name__ == '__main__':
    main()
