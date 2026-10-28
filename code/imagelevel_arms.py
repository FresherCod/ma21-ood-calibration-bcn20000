# -*- coding: utf-8 -*-
"""
Bốn nhánh đối chứng để tách tác động của phép biến đổi (placebo) khỏi tín hiệu far-OOD.

  orig     ảnh ID test gốc (đối chiếu: phải khớp AUROC gốc của imagelevel_control)
  colour   ID đã khớp màu và cường độ với far (histogram matching), không làm mờ, không làm sắc nét
  full     ID đã khớp màu rồi áp dụng phép biến đổi độ dốc (như imagelevel_control)
  placebo  ID gốc chỉ áp dụng cùng phép biến đổi độ dốc của nhánh full, KHÔNG khớp màu, KHÔNG nhắm vào far
  colour_placebo  ID đã khớp màu với MỘT TẬP ID KHÁC (cùng miền, không phải far): đo tác động của bản thân phép
           khớp histogram khi đích không phải far

Với mỗi nhánh: AUROC so với far gốc (vs_far) và AUROC so với ID gốc (shift). Chỉ dùng năng lượng và MSP
sau nhiệt độ T (không dùng Mahalanobis/kNN để chạy nhanh). Mẫu ID và far được chọn theo đúng thứ tự RNG
của imagelevel_control, nên nhánh orig phải khớp với lần chạy trước.

Đọc kết quả (quy tắc đặt trước):
  - shift của placebo gần shift của full: phép biến đổi độ dốc tự đổi điểm số, không phải khớp màu.
  - nhánh colour: diễn giải được nếu shift nằm trong 0,5 ± 0,05 và |SMD độ bão hòa| <= 0,2.
    Khi đó CI của vs_far chứa 0,5 nghĩa là màu và cường độ đủ giải thích; không chứa 0,5 nghĩa là tín hiệu vượt màu.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_calibration as m  # noqa: E402
import imagelevel_control as ic  # noqa: E402

OUT = os.environ.get('OUT', os.path.join(m.RUN_ROOT, 'imagelevel_arms'))
ARMS = ['orig', 'colour', 'full', 'placebo', 'colour_placebo']
ARMS_RUN = [a for a in os.environ.get('ARMS', ','.join(ARMS)).split(',') if a]
SCORES = ['energy_ts', 'msp_ts']


def apply_transform(imgs, kind, param):
    """Áp dụng đúng phép biến đổi độ dốc đã chọn (không khớp màu)."""
    if kind == 'none':
        return imgs
    if kind == 'blur':
        return ic._apply_chunked(imgs, lambda x: gaussian_filter(x, sigma=(0, param, param, 0)))
    if kind == 'sharpen':
        return ic._apply_chunked(imgs, lambda x: x + param * (x - gaussian_filter(x, sigma=(0, 1.0, 1.0, 0))))
    raise ValueError(kind)


def logit_scores(logits, T):
    return {'energy_ts': (T * torch.logsumexp(logits / T, 1)).numpy(),
            'msp_ts': torch.softmax(logits / T, 1).max(1).values.numpy()}


def auroc_pair(a, b):
    """AUROC khi a là lớp dương (giống ID) và b là lớp âm."""
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return float(roc_auc_score(y, np.r_[a, b]))


def run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp):
    t0 = time.time()
    S = m.build_split(seed, rowmap)
    K = len(S['classes'])
    y, row_id = S['y'], S['idd']['row'].values
    rng = np.random.default_rng(seed + 100)            # cùng thứ tự RNG với imagelevel_control khi N_BANK=0

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
    s_id = logit_scores(L_id, T)
    sat_id, grad_id = m.lowlevel_scores(id_imgs, np.arange(n))

    out = {'seed': seed, 'T': T, 'n_id': n, 'sources': {}}
    for si, src in enumerate(np.unique(far_src)):
        mk = np.flatnonzero(far_src == src)
        pick = rng.choice(mk, min(ic.N_FAR_SUB, len(mk)), replace=False)
        far_imgs = np.asarray(Xfar[pick])
        nf = len(far_imgs)
        _, L_far, _ = m.get_feats_logits(model, far_imgs, np.arange(nf), np.full(nf, -1), device, use_amp)
        s_far = logit_scores(L_far, T)
        sat_far, grad_far = m.lowlevel_scores(far_imgs, np.arange(nf))

        colour = ic.color_match(id_imgs, far_imgs)
        target = float(np.median(grad_far))
        full, kind, param, g_full = ic.transform_to_grad(colour, target)
        placebo = apply_transform(id_imgs, kind, param)
        # tập ID khác để làm đích cho placebo màu: RNG riêng, không làm lệch thứ tự chọn mẫu
        rng2 = np.random.default_rng(seed + 200 + si)
        other = np.setdiff1d(te, te_sub)
        other_rows = row_id[rng2.choice(other, min(ic.N_ID_SUB, len(other)), replace=False)]
        other_imgs = np.asarray(X[other_rows])
        colour_placebo = ic.color_match(id_imgs, other_imgs)
        imgs = {'orig': id_imgs, 'colour': colour, 'full': full, 'placebo': placebo,
                'colour_placebo': colour_placebo}

        rec = {'n_far': nf, 'gray_set': ic.is_gray_set(far_imgs), 'transform': {'kind': kind, 'param': param},
               'grad_target': target, 'grad_full': g_full, 'arms': {}}
        for arm in ARMS:
            if arm != 'orig' and arm not in ARMS_RUN:
                continue
            if arm == 'orig':
                s_arm = s_id
            else:
                _, L_arm, _ = m.get_feats_logits(model, imgs[arm], np.arange(n), np.full(n, -1), device, use_amp)
                s_arm = logit_scores(L_arm, T)
            sat_a, grad_a = m.lowlevel_scores(imgs[arm], np.arange(n))
            rec['arms'][arm] = {
                'balance_sat_smd': ic.smd(sat_a, sat_far),
                'balance_grad_smd': ic.smd(grad_a, grad_far),
                'auc': {k: {'vs_far': auroc_pair(s_arm[k], s_far[k]),
                            'shift': auroc_pair(s_id[k], s_arm[k])} for k in SCORES},
            }
        out['sources'][str(src)] = rec
        parts = [f'{arm} {v["auc"]["energy_ts"]["vs_far"]:.3f} (shift {v["auc"]["energy_ts"]["shift"]:.3f})'
                 for arm, v in rec['arms'].items()]
        print(f'  seed {seed} {src}: ' + ' | '.join(parts) + f' | {kind}({param})', flush=True)
    out['seconds'] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'arms_seed{seed}.json'), 'w', encoding='utf-8') as f:
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


def load_arms(out_dir):
    rows = []
    for p in sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith('arms_seed')):
        with open(p, encoding='utf-8') as f:
            r = json.load(f)
        for src, d in r['sources'].items():
            for arm, a in d['arms'].items():
                for k in SCORES:
                    rows.append(dict(seed=r['seed'], source=src, arm=arm, method=k,
                                     vs_far=a['auc'][k]['vs_far'], shift=a['auc'][k]['shift'],
                                     sat_smd=a['balance_sat_smd'], grad_smd=a['balance_grad_smd'],
                                     transform=f"{d['transform']['kind']}({d['transform']['param']})",
                                     gray_set=d['gray_set']))
    return pd.DataFrame(rows)


def verdict_colour(n, ci_lo, ci_hi, shift_mu, sat_smd_mu):
    if n < 2 or np.isnan(ci_lo):
        return 'không kết luận: cần ít nhất 2 seed'
    if abs(shift_mu - 0.5) > 0.05:
        return 'không kết luận: khớp màu tự đổi điểm (shift lệch 0,5)'
    if sat_smd_mu is None or np.isnan(sat_smd_mu) or abs(sat_smd_mu) > 0.2:
        return 'không kết luận: cân bằng độ bão hòa kém'
    if ci_lo <= 0.5 <= ci_hi:
        return 'màu và cường độ đủ giải thích tín hiệu (CI chứa 0,5)'
    return 'tín hiệu vượt màu và cường độ'


def summarize(dirs):
    frames = []
    for d in dirs:
        label, path = d.split('=', 1) if '=' in d else (os.path.basename(os.path.normpath(d)), d)
        df = load_arms(path)
        if not df.empty:
            df.insert(0, 'run', label)
            frames.append(df)
    if not frames:
        raise SystemExit('Không có arms_seed*.json.')
    all_df = pd.concat(frames, ignore_index=True)
    # cùng một (nhãn, seed, nguồn, nhánh) chỉ được tính một lần, kể cả khi nhánh gốc xuất hiện ở nhiều thư mục
    all_df = all_df.drop_duplicates(subset=['run', 'seed', 'source', 'arm', 'method'])
    all_df.to_csv(os.path.join(OUT, 'arms_by_seed.csv'), index=False)
    rows = []
    for (run, src, arm, meth), g in all_df.groupby(['run', 'source', 'arm', 'method']):
        v = t_ci(g['vs_far']); sh = t_ci(g['shift']); ss = t_ci(g['sat_smd'])[0]; sg = t_ci(g['grad_smd'])[0]
        rows.append(dict(run=run, source=src, arm=arm, method=meth, n=int(g['seed'].nunique()),
                         transform=g['transform'].iloc[0], vs_far=v[0], vs_far_ci=f'[{v[1]:.3f}; {v[2]:.3f}]',
                         shift=sh[0], shift_ci=f'[{sh[1]:.3f}; {sh[2]:.3f}]', sat_smd=ss, grad_smd=sg,
                         verdict=verdict_colour(int(g['seed'].nunique()), v[1], v[2], sh[0], ss)
                         if (arm == 'colour' and meth == 'energy_ts') else ''))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, 'arms_summary.csv'), index=False)
    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', 30)
    print(summ.round(3).to_string(index=False))


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
    print('device:', device, '| BACKBONE:', m.BACKBONE, '| arms:', ARMS, flush=True)
    for seed in [int(s) for s in m.SEEDS]:
        run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp)


if __name__ == '__main__':
    main()
