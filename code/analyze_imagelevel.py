# -*- coding: utf-8 -*-
"""
Tổng hợp đối chứng ở mức ảnh (imagelevel_seed*.json) từ một hoặc nhiều thư mục OUT.

Với mỗi run, bộ far và phương pháp, báo cáo trung bình và CI t 95% theo seed của:
  orig      AUROC ID gốc so với far gốc
  matched   AUROC ID đã biến đổi so với far gốc (đối chứng chính)
  shift     AUROC ID gốc so với ID đã biến đổi (tác động của chính phép biến đổi)
  transform loại và tham số biến đổi đã chọn (làm mờ hoặc làm sắc nét); smd_sat, smd_grad: cân bằng sau biến đổi

Quy tắc đọc (đặt trước khi xem số liệu):
  - matched CI bao gồm 0,5 và smd_sat, smd_grad đều trong ±0,2: tín hiệu nằm trong các thống kê đã khớp.
  - matched CI không chứa 0,5 và cân bằng tốt: tín hiệu vượt quá các thống kê đã khớp (chưa loại trừ mọi đường tắt).
  - cân bằng kém: không kết luận được, bất kể AUROC.
  - ít hơn 2 seed: không kết luận (không có CI).
  - shift lệch khỏi 0,5 hơn 0,05: cảnh báo, phép biến đổi tự đổi điểm số.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ['energy_ts', 'msp_ts', 'maha', 'knn']


def t_ci(x):
    x = np.asarray([v for v in x if v is not None], float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, np.nan
    sd = float(x.std(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - h, m + h


def load(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, 'imagelevel_seed*.json'))):
        with open(p, encoding='utf-8') as f:
            r = json.load(f)
        for src, d in r['sources'].items():
            for k in METHODS:
                a = d['auc'][k]
                rows.append(dict(seed=r['seed'], source=src, method=k, orig=a['orig'], matched=a['matched'],
                                 shift=a['shift'], t_kind=d['transform']['kind'], t_param=d['transform']['param'],
                                 gray_set=d['gray_set'], smd_sat=d['balance_sat_smd'], smd_grad=d['balance_grad_smd'],
                                 grad_after=d['grad_after'], grad_target=d['grad_target'], n_far=d['n_far']))
    return pd.DataFrame(rows)


def verdict(n_seeds, matched_ci, shift_mean, smd_sat, smd_grad):
    """Kết luận theo quy tắc đọc đã ghi trước khi xem số liệu (xem docstring đầu file)."""
    lo, hi = matched_ci
    if n_seeds < 2 or np.isnan(lo) or np.isnan(hi):
        return 'không kết luận: cần ít nhất 2 seed để có CI'
    bal = all(v is not None and not np.isnan(v) and abs(v) <= 0.2 for v in (smd_sat, smd_grad))
    if not bal:
        return 'không kết luận: cân bằng kém'
    note = ''
    if not np.isnan(shift_mean) and abs(shift_mean - 0.5) > 0.05:
        note = f'; cảnh báo: phép biến đổi tự đổi điểm (shift {shift_mean:.2f})'
    if lo <= 0.5 <= hi:
        return 'tín hiệu nằm trong thống kê đã khớp' + note
    return 'tín hiệu vượt thống kê đã khớp' + note


def main():
    ap = argparse.ArgumentParser(description='Tổng hợp đối chứng ở mức ảnh')
    ap.add_argument('dirs', nargs='+', help='thư mục OUT chứa imagelevel_seed*.json (đặt tên theo nhãn: tên=đường_dẫn)')
    ap.add_argument('--out', default='analysis/imagelevel')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = []
    for d in args.dirs:
        label, path = (d.split('=', 1) if '=' in d else (os.path.basename(os.path.normpath(d)), d))
        df = load(path)
        if df.empty:
            print(f'{label}: không có dữ liệu trong {path}')
            continue
        df.insert(0, 'run', label)
        frames.append(df)
    if not frames:
        raise SystemExit('Không có dữ liệu.')
    all_df = pd.concat(frames, ignore_index=True)

    summ = []
    for (run, src, meth), g in all_df.groupby(['run', 'source', 'method']):
        o = t_ci(g['orig']); mt = t_ci(g['matched']); sh = t_ci(g['shift'])
        ss = t_ci(g['smd_sat'])[0]; sg = t_ci(g['smd_grad'])[0]
        n_seeds = int(g['seed'].nunique())
        summ.append(dict(run=run, source=src, method=meth, n_seeds=n_seeds,
                         gray_set=bool(g['gray_set'].iloc[0]), orig=o[0], orig_ci=f'[{o[2]:.3f}; {o[3]:.3f}]',
                         matched=mt[0], matched_ci=f'[{mt[2]:.3f}; {mt[3]:.3f}]',
                         shift=sh[0], transform=f"{g['t_kind'].iloc[0]}({g['t_param'].iloc[0]})",
                         smd_sat=ss, smd_grad=sg,
                         verdict=verdict(n_seeds, (mt[2], mt[3]), sh[0], ss, sg) if meth == 'energy_ts' else ''))
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(os.path.join(args.out, 'imagelevel_summary.csv'), index=False)
    all_df.to_csv(os.path.join(args.out, 'imagelevel_by_seed.csv'), index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(summ_df.round(3).to_string(index=False))
    print(f'\nĐã ghi vào {args.out}/')


if __name__ == '__main__':
    main()
