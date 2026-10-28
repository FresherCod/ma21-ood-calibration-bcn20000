# -*- coding: utf-8 -*-
"""
Phân tích A3 (mục 14, phiên bản 2): can thiệp bất biến độ sáng và tương phản (J) so với đối chứng (B).

Đầu vào: thư mục outputs của J và B (results_seed*.json, predictions_seed*.csv, far_scores_seed*.csv),
tập far con (farsub), CSV train, và X_train_256.u8 (cho kiểm K1 theo bin độ sáng, không huấn luyện).
Đầu ra: bảng quyết định (CSV), bảng K1 (CSV) và tóm tắt in ra màn hình.

Dùng: python code/analyze_jit.py --j DIR_J --b DIR_B --out analysis/jit_a3
      thêm --smoke để chạy với ít seed (không tính CI có ý nghĩa).
Hàm gates và decision_rows được analyze_e.py dùng lại cho E3 và E5 (mục 16).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

FAR_DIR = 'C:/data/bcn20000/farsub'
CSV = 'D:/NCKH_3/ma21_ood_calibration_bcn20000/data/bcn_20k_train.csv'
X_PATH = 'C:/data/bcn20000/X_train_256.u8'
RESIZE, CROP = 256, 224
OFF = (RESIZE - CROP) // 2
PRIMARY = ['retinamnist_224', 'pathmnist_224']          # bộ chính (mục 14)
REPORT_ONLY = ['bloodmnist_224', 'breastmnist_224', 'pneumoniamnist_224']
FAR_ORDER = ['retinamnist_224', 'pneumoniamnist_224', 'breastmnist_224', 'bloodmnist_224', 'pathmnist_224']


def load(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, 'results_seed*.json')), key=lambda s: int(s.split('seed')[-1].split('.')[0])):
        with open(p, encoding='utf-8') as f:
            out.append(json.load(f))
    return out


def t_ci(x, level=0.9875):
    """CI t hai phía. level là phân vị của t: 0,9875 cho CI hai phía 97,5% (quy ước mục 14 và 16)."""
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(level, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def auc_source(R, src):
    return [r['far_ood'][src]['energy_ts']['auroc'] for r in R if src in r.get('far_ood', {})]


def lum_model_frame(arr):
    """Luminance trung bình của far theo đúng khung mô hình: phóng 224 -> 256 bilinear, cắt giữa 224."""
    out = np.empty(len(arr), dtype=np.float32)
    for i, a in enumerate(arr):
        rgb = np.stack([a] * 3, -1) if a.ndim == 2 else a.astype(np.uint8)
        im = Image.fromarray(rgb.astype(np.uint8)).convert('RGB').resize((RESIZE, RESIZE), Image.BILINEAR)
        px = np.asarray(im, dtype=np.float32)[OFF:OFF + CROP, OFF:OFF + CROP]
        out[i] = (0.299 * px[..., 0] + 0.587 * px[..., 1] + 0.114 * px[..., 2]).mean()
    return out


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def k1_bins(out_dir, seeds, far_lum, rowmap, X, n_bins=5, min_per_bin=20):
    """Mục 14, K1: AUROC năng lượng trong từng bin độ sáng (bin theo phân vị của gộp ID và far), trung bình có trọng số."""
    rows = []
    for s in seeds:
        pred = pd.read_csv(f'{out_dir}/predictions_seed{s}.csv')
        ide = pred.loc[pred['is_ood'] == 0, ['bcn_filename', 'energy_ts']]
        idx = np.array([rowmap[n] for n in ide['bcn_filename']])
        id_lum = np.asarray(X[np.sort(idx)][:, OFF:OFF + CROP, OFF:OFF + CROP]).astype(np.float32)
        id_lum = (0.299 * id_lum[..., 0] + 0.587 * id_lum[..., 1] + 0.114 * id_lum[..., 2]).mean(axis=(1, 2))
        order = np.argsort(idx)
        id_e = ide['energy_ts'].to_numpy()[order]
        far = pd.read_csv(f'{out_dir}/far_scores_seed{s}.csv')
        for src in FAR_ORDER:
            fe = far.loc[far['source'] == src, 'energy_ts'].to_numpy()
            fl = far_lum[src]
            if len(fe) != len(fl):
                raise RuntimeError(f'{src}: far_scores có {len(fe)} dòng, npz có {len(fl)} ảnh')
            lum = np.r_[id_lum, fl]
            en = np.r_[id_e, fe]
            is_id = np.r_[np.ones(len(id_lum), bool), np.zeros(len(fl), bool)]
            edges = np.quantile(lum, np.linspace(0, 1, n_bins + 1))
            b = np.clip(np.searchsorted(edges, lum, side='right') - 1, 0, n_bins - 1)
            for k in range(n_bins):
                m = b == k
                ni, nf = int((m & is_id).sum()), int((m & ~is_id).sum())
                if ni < min_per_bin or nf < min_per_bin:
                    rows.append(dict(seed=s, source=src, bin=k, n_id=ni, n_far=nf, auroc=np.nan))
                    continue
                a = auroc(en[m & is_id], en[m & ~is_id])
                rows.append(dict(seed=s, source=src, bin=k, n_id=ni, n_far=nf, auroc=a))
    return pd.DataFrame(rows)


def gates(RJ, RB):
    """Cổng G1, G2, G3 của mục 14: nhánh J (can thiệp) so với B (đối chứng). C0 tính trong decision_rows."""
    ba_j = [r['id_metrics_ts']['bal_acc'] for r in RJ]
    ba_b = [r['id_metrics_ts']['bal_acc'] for r in RB]
    ak_j = [r['ood']['energy_ts']['auroc'] for r in RJ]
    ak_b = [r['ood']['energy_ts']['auroc'] for r in RB]
    paired = {r['seed']: r['id_metrics_ts']['bal_acc'] for r in RB}
    diffs = [r['id_metrics_ts']['bal_acc'] - paired[r['seed']] for r in RJ if r['seed'] in paired]
    g1 = bool((np.mean(diffs) >= -0.03) and (np.min(diffs) >= -0.05))
    g2 = bool(np.mean(ak_j) >= np.mean(ak_b) - 0.05)
    d1_j = [r['brightness_sens']['d1_norm'] for r in RJ if 'brightness_sens' in r]
    d1_b = [r['brightness_sens']['d1_norm'] for r in RB if 'brightness_sens' in r]
    s_ratio = (np.mean(d1_j) / np.mean(d1_b)) if d1_j and d1_b else np.nan
    g3 = bool(s_ratio <= 0.5) if not np.isnan(s_ratio) else None      # None: không có D1 để tính S
    return dict(g1=g1, g2=g2, g3=g3, ba_j=ba_j, ba_b=ba_b, ak_j=ak_j, ak_b=ak_b, ba_diff=diffs, s_ratio=s_ratio)


def decision_rows(RJ, RB, g, sources):
    """Bảng quyết định mục 14 cho từng bộ far: AUROC, CI, C0, R và quyết định (bộ chính có cổng)."""
    rows = []
    for src in sources:
        aj_ = auc_source(RJ, src)
        ab_ = auc_source(RB, src)
        if len(aj_) < 2 or len(ab_) < 2:
            rows.append(dict(source=src, note='thiếu dữ liệu'))
            continue
        mj, lj, hj = t_ci(aj_, 0.9875)            # Bonferroni cho 2 bộ chính (97,5% mỗi bộ)
        mb, lb, hb = t_ci(ab_, 0.9875)
        c0 = lb > 0.6
        # Cách chính (mục 12, "như mục 12"): mẫu số là điểm ước lượng của B; chỉ các cận của J đổi.
        r_mid = (mj - 0.5) / (mb - 0.5)
        r_lo = (lj - 0.5) / (mb - 0.5)
        r_hi = (hj - 0.5) / (mb - 0.5)
        # Độ nhạy (chỉ báo cáo): đọc "cận của A_J và A_B" theo nghĩa cả hai cận cùng đổi.
        if src in PRIMARY:
            # Cực trị trên hình hộp (cận A_J) × (cận A_B): tỉ số đơn điệu theo từng biến khi mẫu số dương, nên cực trị ở đỉnh.
            corners = [(x - 0.5) / (y - 0.5) for x in (lj, hj) for y in (lb, hb)]
            r_lo_both, r_hi_both = min(corners), max(corners)
        else:
            r_lo_both = r_hi_both = np.nan
        gate_ok = dict(C0=c0, G1=g['g1'], G2=g['g2'], G3=g['g3'])
        if src in PRIMARY:
            if g['g3'] is None:
                dec = 'không kết luận: thiếu D1, không tính được G3'
            elif not all(gate_ok.values()):
                failed = [k for k, v in gate_ok.items() if not v]
                dec = 'không kết luận: không đạt ' + ', '.join(failed)
            elif r_lo >= 0.8:
                dec = 'còn tín hiệu sau bất biến độ sáng và tương phản (R cận dưới ≥ 0,8)'
            elif r_hi <= 0.2:
                dec = 'giảm mạnh khi bất biến độ sáng và tương phản (R cận trên ≤ 0,2)'
            else:
                dec = 'không kết luận: CI của R không nằm hoàn toàn trên 0,8 hoặc dưới 0,2'
        else:
            dec = 'chỉ báo cáo (không phải bộ chính)'
        sd_j, sd_b = np.std(aj_, ddof=1), np.std(ab_, ddof=1)
        nj, nb = len(aj_), len(ab_)
        df = nj + nb - 2                          # 10 seed mỗi nhánh cho df = 18, như mục 14
        mde = (stats.t.ppf(0.9875, df) + stats.t.ppf(0.8, df)) * np.sqrt(sd_j ** 2 / nj + sd_b ** 2 / nb)   # cùng CI 97,5% hai phía
        rows.append(dict(source=src, A_J=mj, A_J_lo=lj, A_J_hi=hj, A_B=mb, A_B_lo=lb, A_B_hi=hb,
                         C0_signal_in_B=c0, R=r_mid, R_lo=r_lo, R_hi=r_hi,
                         R_lo_both_bounds=r_lo_both, R_hi_both_bounds=r_hi_both, MDE_approx=mde, decision=dec))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--j', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', default='analysis/jit_a3')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    RJ, RB = load(a.j), load(a.b)
    if not a.smoke and (len(RJ) < 10 or len(RB) < 10):
        raise SystemExit('Mục 14 yêu cầu 10 seed mỗi nhánh. Dùng --smoke chỉ để thử.')
    if not a.smoke:
        for R, name in [(RJ, 'J'), (RB, 'B')]:
            if sorted(r['seed'] for r in R) != list(range(10)):
                raise SystemExit(f'{name}: mục 14 yêu cầu đúng seed 0 đến 9, có {sorted(r["seed"] for r in R)}')

    # --- G1, G2, G3 và C0 ---
    g = gates(RJ, RB)
    diffs = g['ba_diff']
    print(f'nhánh J: n={len(RJ)}, BA {np.mean(g["ba_j"]):.4f}, AK {np.mean(g["ak_j"]):.4f}')
    print(f'nhánh B: n={len(RB)}, BA {np.mean(g["ba_b"]):.4f}, AK {np.mean(g["ak_b"]):.4f}')
    print(f'G1 (BA hiệu trung bình {np.mean(diffs):+.4f}, nhỏ nhất {np.min(diffs):+.4f}): {"đạt" if g["g1"] else "KHÔNG đạt"}')
    print(f'G2 (AK_J {np.mean(g["ak_j"]):.4f} ≥ AK_B − 0,05 = {np.mean(g["ak_b"]) - 0.05:.4f}): {"đạt" if g["g2"] else "KHÔNG đạt"}')
    g3_txt = 'thiếu D1' if g['g3'] is None else ('đạt' if g['g3'] else 'KHÔNG đạt')
    print(f'G3 (S = D1_J / D1_B = {g["s_ratio"]:.4f} ≤ 0,5): {g3_txt}')

    # --- bộ chính: C0, R, quyết định ---
    dec_df = pd.DataFrame(decision_rows(RJ, RB, g, PRIMARY + REPORT_ONLY))
    dec_df.to_csv(os.path.join(a.out, 'decision.csv'), index=False)
    pd.set_option('display.width', 240)
    pd.set_option('display.max_colwidth', 80)
    print(dec_df.round(3).to_string(index=False))
    if a.smoke:
        print('\n(smoke: bỏ qua K1, vì tên ảnh của bộ thử không có trong CSV thật)')
        return

    # --- K1: kiểm không huấn luyện theo bin độ sáng ---
    far_lum = {src: lum_model_frame(np.load(f'{FAR_DIR}/{src}.npz')['test_images']) for src in FAR_ORDER}
    names = pd.read_csv(CSV)
    if 'split' in names.columns:
        names = names[names['split'].astype(str).str.lower() == 'train']
    rowmap = {n: i for i, n in enumerate(names['bcn_filename'].tolist())}
    X = np.memmap(X_PATH, dtype=np.uint8, mode='r', shape=(len(rowmap), RESIZE, RESIZE, 3))
    seeds_j = [r['seed'] for r in RJ]
    seeds_b = [r['seed'] for r in RB]
    k1 = []
    for tag, d, seeds in [('J', a.j, seeds_j), ('B', a.b, seeds_b)]:
        kb = k1_bins(d, seeds, far_lum, rowmap, X)
        kb['model'] = tag
        k1.append(kb)
    k1 = pd.concat(k1, ignore_index=True)
    k1.to_csv(os.path.join(a.out, 'k1_bins.csv'), index=False)
    summ = []
    for (tag, src), grp in k1.groupby(['model', 'source']):
        ok = grp.dropna(subset=['auroc'])
        if len(ok):
            w = ok['n_id'] + ok['n_far']
            avg = float(np.average(ok['auroc'], weights=w))
            summ.append(dict(model=tag, source=src, bins_used=len(ok), auroc_within_bins=avg, gt_0_6=avg > 0.6))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(a.out, 'k1_summary.csv'), index=False)
    print('\nK1, AUROC trong bin độ sáng (có trọng số, bin có ít nhất 20 ảnh mỗi nhóm):')
    print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
