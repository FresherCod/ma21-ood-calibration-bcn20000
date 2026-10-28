# -*- coding: utf-8 -*-
"""
Đối chứng khử màu hai phía (docs/dose_response_design.md mục 10).

Với mỗi seed và bộ far:
  orig       ID màu so với far màu (đối chiếu với kết quả chính)
  gg         ID xám so với far xám (luminance, ba kênh giống nhau)
  gg_lum     ID xám đã khớp histogram độ sáng với far xám so với far xám
Chấm bằng năng lượng và MSP sau nhiệt độ T. Mẫu ID và far chọn đúng thứ tự RNG của imagelevel_dose.
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
import imagelevel_dose as dz  # noqa: E402

OUT = os.environ.get('OUT', os.path.join(m.RUN_ROOT, 'imagelevel_graycontrol'))
SCORES = ['energy_ts', 'msp_ts']


def to_gray3(imgs):
    lum = np.round(0.299 * imgs[..., 0] + 0.587 * imgs[..., 1] + 0.114 * imgs[..., 2]).clip(0, 255).astype(np.uint8)
    return np.repeat(lum[..., None], 3, axis=-1)


def run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp):
    t0 = time.time()
    S = m.build_split(seed, rowmap)
    K = len(S['classes'])
    row_id = S['idd']['row'].values
    rng = np.random.default_rng(seed + 100)
    with open(os.path.join(ic.RESULTS_DIR, f'results_seed{seed}.json'), encoding='utf-8') as f:
        T = float(json.load(f)['temperature'])
    model = m.build_model(K).to(device)
    ck = torch.load(os.path.join(ic.CKPT_DIR, f'seed{seed}', 'best_model.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state'])
    model.eval()

    def score(imgs):
        _, L, _ = m.get_feats_logits(model, imgs, np.arange(len(imgs)), np.full(len(imgs), -1), device, use_amp)
        return ia.logit_scores(L, T)

    te = S['te']
    te_sub = rng.choice(te, min(ic.N_ID_SUB, len(te)), replace=False)
    id_imgs = np.asarray(X[row_id[te_sub]])
    n = len(id_imgs)
    s_id = score(id_imgs)
    id_gray = to_gray3(id_imgs)
    s_idg = score(id_gray)

    out = {'seed': seed, 'T': T, 'n_id': n, 'sources': {}}
    for src in np.unique(far_src):
        mk = np.flatnonzero(far_src == src)
        pick = rng.choice(mk, min(ic.N_FAR_SUB, len(mk)), replace=False)
        far_imgs = np.asarray(Xfar[pick])
        nf = len(far_imgs)
        s_far = score(far_imgs)
        far_gray = to_gray3(far_imgs)
        s_farg = score(far_gray)
        lum_matched = ic.color_match(id_gray, far_gray)
        s_lm = score(lum_matched)
        lum_far = dz.lum_of(far_gray)
        lum_id_g = dz.lum_of(id_gray)
        lum_lm = dz.lum_of(lum_matched)
        rec = {'n_far': nf, 'gray_set': ic.is_gray_set(far_imgs), 'arms': {}}
        rec['arms']['orig'] = {'auc': {k: ia.auroc_pair(s_id[k], s_far[k]) for k in SCORES}}
        rec['arms']['gg'] = {'auc': {k: ia.auroc_pair(s_idg[k], s_farg[k]) for k in SCORES},
                             'smd_lum': ic.smd(lum_id_g, lum_far)}
        rec['arms']['gg_lum'] = {'auc': {k: ia.auroc_pair(s_lm[k], s_farg[k]) for k in SCORES},
                                 'smd_lum': ic.smd(lum_lm, lum_far)}
        # shift: đo mức đổi điểm do chính phép khử màu (orig là lớp dương). ~0,5 nghĩa là không đổi điểm.
        rec['shift'] = {'id': {k: ia.auroc_pair(s_id[k], s_idg[k]) for k in SCORES},
                        'far': {k: ia.auroc_pair(s_far[k], s_farg[k]) for k in SCORES}}
        out['sources'][str(src)] = rec
        print(f'  seed {seed} {src}: orig {rec["arms"]["orig"]["auc"]["energy_ts"]:.3f} | '
              f'gg {rec["arms"]["gg"]["auc"]["energy_ts"]:.3f} | gg_lum {rec["arms"]["gg_lum"]["auc"]["energy_ts"]:.3f} | '
              f'SMD độ sáng gg {rec["arms"]["gg"]["smd_lum"]:.3f}', flush=True)
    out['seconds'] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'graycontrol_seed{seed}.json'), 'w', encoding='utf-8') as f:
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


RET_HI = 0.8   # giữ ≥ 80% tín hiệu gốc: không phụ thuộc màu
RET_LO = 0.2   # giữ ≤ 20% tín hiệu gốc: phụ thuộc màu
SHIFT_BAND = (0.45, 0.55)


def decide(orig_ci, shift_id_ci, ret_gg_ci, ret_gl_ci, gray_set):
    """Quy tắc đặt trước (docs/dose_response_design.md mục 10, v0.4). Thứ tự ưu tiên từ trên xuống."""
    if gray_set:
        return 'không áp dụng: bộ far đã là thang xám'
    if np.isnan(ret_gg_ci[0]):
        return 'không kết luận: cần ít nhất 2 seed'
    if orig_ci[0] <= 0.5 <= orig_ci[1]:
        return 'không có tín hiệu gốc'
    if not (SHIFT_BAND[0] <= shift_id_ci[0] and shift_id_ci[1] <= SHIFT_BAND[1]):
        return 'không kết luận: bỏ màu tự đổi điểm ID (shift ngoài 0,5 ± 0,05)'
    if ret_gg_ci[0] >= RET_HI:
        return 'không phụ thuộc màu (giữ ≥ 80% tín hiệu khi bỏ màu)'
    if ret_gg_ci[1] <= RET_LO:
        if ret_gl_ci[0] >= RET_HI:
            return 'phụ thuộc màu (mất ≥ 80%) nhưng giữ khi khớp độ sáng: phụ thuộc độ sáng'
        return 'phụ thuộc màu (mất ≥ 80% tín hiệu khi bỏ màu)'
    return 'không kết luận: giữ một phần tín hiệu (20–80%)'


def summarize(out_dir):
    rows, per_seed = [], []
    for f in sorted(os.listdir(out_dir)):
        if not f.startswith('graycontrol_seed'):
            continue
        r = json.load(open(os.path.join(out_dir, f), encoding='utf-8'))
        for src, d in r['sources'].items():
            for arm, a in d['arms'].items():
                for k in SCORES:
                    rows.append(dict(seed=r['seed'], source=src, arm=arm, method=k, gray_set=d['gray_set'],
                                     auc=a['auc'][k], smd_lum=a.get('smd_lum')))
            if 'shift' in d:
                for k in SCORES:
                    per_seed.append(dict(seed=r['seed'], source=src, method=k, gray_set=d['gray_set'],
                                         shift_id=d['shift']['id'][k], shift_far=d['shift']['far'][k],
                                         orig=d['arms']['orig']['auc'][k], gg=d['arms']['gg']['auc'][k],
                                         gg_lum=d['arms']['gg_lum']['auc'][k]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'graycontrol_by_seed.csv'), index=False)
    ps = pd.DataFrame(per_seed)
    ps.to_csv(os.path.join(out_dir, 'graycontrol_per_seed_shift.csv'), index=False)
    res = []
    for src, g in ps[ps.method == 'energy_ts'].groupby('source'):
        g = g.sort_values('seed')
        orig_c = t_ci(g['orig'].values)
        shift_c = t_ci(g['shift_id'].values)
        # tỉ lệ giữ tín hiệu theo từng seed: (A − 0,5)/(A_gốc − 0,5), chỉ khi tín hiệu gốc của seed dương
        def ret(col):
            num = g[col].values - 0.5
            den = g['orig'].values - 0.5
            return t_ci(num / np.where(np.abs(den) < 1e-9, np.nan, den))
        ret_gg, ret_gl = ret('gg'), ret('gg_lum')
        gray = bool(g['gray_set'].iloc[0])
        res.append(dict(source=src, gray=gray, n=int(g['seed'].nunique()),
                        orig=orig_c[0], orig_ci=f'[{orig_c[1]:.3f}; {orig_c[2]:.3f}]',
                        shift_id=shift_c[0], shift_id_ci=f'[{shift_c[1]:.3f}; {shift_c[2]:.3f}]',
                        shift_far=t_ci(g['shift_far'].values)[0],
                        gg=t_ci(g['gg'].values)[0], retain_gg=ret_gg[0],
                        retain_gg_ci=f'[{ret_gg[1]:.3f}; {ret_gg[2]:.3f}]',
                        retain_gg_lum=ret_gl[0], retain_gl_ci=f'[{ret_gl[1]:.3f}; {ret_gl[2]:.3f}]',
                        decision=decide((orig_c[1], orig_c[2]), (shift_c[1], shift_c[2]),
                                        (ret_gg[1], ret_gg[2]), (ret_gl[1], ret_gl[2]), gray)))
    summ = pd.DataFrame(res)
    summ.to_csv(os.path.join(out_dir, 'graycontrol_summary.csv'), index=False)
    pd.set_option('display.width', 280)
    pd.set_option('display.max_columns', 30)
    print(summ.round(3).to_string(index=False))


def main():
    if os.environ.get('SUMMARIZE_DIR'):
        summarize(os.environ['SUMMARIZE_DIR'])
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
    print('device:', device, '| BACKBONE:', m.BACKBONE, flush=True)
    for seed in [int(s) for s in m.SEEDS]:
        run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp)


if __name__ == '__main__':
    main()
