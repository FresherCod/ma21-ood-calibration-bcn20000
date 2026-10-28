# -*- coding: utf-8 -*-
"""
Tổng hợp đợt huấn luyện xám (v5g) theo quy tắc đặt trước v0.5 (docs/dose_response_design.md mục 12).

Đầu vào:
  --gray  thư mục output xám chứa results_seed*.json (10 seed)
  --color thư mục output màu v3 chứa results_seed*.json (5 seed; BA và AK tham chiếu)
  --dose  thư mục dose_v3 chứa dose_seed*.json (5 seed; AUROC gốc của cả năm bộ far)
Đầu ra: bảng in ra màn hình và graycolor_decision.csv trong thư mục --out.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

BOUNDS = {'retinamnist_224': 'retinamnist_224', 'pathmnist_224': 'pathmnist_224', 'bloodmnist_224': 'bloodmnist_224'}
COLOUR_SETS = list(BOUNDS.keys())
GRAY_SETS = ['pneumoniamnist_224', 'breastmnist_224']


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    mu = float(x.mean())
    if n < 2:
        return mu, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return mu, mu - h, mu + h


def load_results(out_dir):
    res = []
    for p in sorted(glob.glob(os.path.join(out_dir, 'results_seed*.json'))):
        with open(p, encoding='utf-8') as f:
            res.append(json.load(f))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gray', required=True)
    ap.add_argument('--color', required=True)
    ap.add_argument('--dose', default=None, help='thư mục dose_v3 (tham chiếu v5g); không dùng khi có --ref-far')
    ap.add_argument('--ref-far', default=None, help='thư mục results của mô hình màu cùng LR (tham chiếu v5h: v3h)')
    ap.add_argument('--out', default='analysis/gray_v5g')
    a = ap.parse_args()
    if a.ref_far is None and a.dose is None:
        raise SystemExit('Cần --dose (v5g) hoặc --ref-far (v5h).')
    os.makedirs(a.out, exist_ok=True)

    gray = load_results(a.gray)
    color = load_results(a.color)
    if len(gray) < 2 or len(color) < 2:
        raise SystemExit('Cần ít nhất 2 seed mỗi bên.')

    # tham chiếu màu: AUROC năng lượng của mô hình màu, trung bình và CI theo seed
    ref = {}
    if a.ref_far is not None:
        # v5h: cùng LR với mô hình xám, cùng tập far con (results_seed*.json của v3h)
        cref = load_results(a.ref_far)
        for src in COLOUR_SETS:
            vals = [r['far_ood'][src]['energy_ts']['auroc'] for r in cref if src in r.get('far_ood', {})]
            ref[src] = t_ci(vals)
    else:
        # v5g: AUROC năng lượng gốc của mô hình màu v3 (nhánh orig của đợt liều)
        doses = []
        for p in sorted(glob.glob(os.path.join(a.dose, 'dose_seed*.json'))):
            with open(p, encoding='utf-8') as f:
                doses.append(json.load(f))
        for src in COLOUR_SETS:
            vals = [d['sources'][src]['arms']['orig']['auc']['energy_ts']['vs_far'] for d in doses]
            ref[src] = t_ci(vals)
    ba_color = t_ci([r['id_metrics_ts']['bal_acc'] for r in color])[0]
    ak_color = t_ci([r['ood']['energy_ts']['auroc'] for r in color])[0]

    ba_gray = t_ci([r['id_metrics_ts']['bal_acc'] for r in gray])
    ak_gray = t_ci([r['ood']['energy_ts']['auroc'] for r in gray])
    gate_model = (ba_gray[0] >= ba_color - 0.05) and (ak_gray[0] >= 0.55)

    rows = []
    for src in COLOUR_SETS + GRAY_SETS:
        vals = [r['far_ood'][src]['energy_ts']['auroc'] for r in gray if src in r.get('far_ood', {})]
        if len(vals) < 2:
            rows.append(dict(source=src, n=len(vals), decision='thiếu dữ liệu: bộ far không có trong kết quả xám'))
            continue
        mu, lo, hi = t_ci(vals)
        lum = [r.get('far_lum_smd', {}).get(src) for r in gray]
        lum = [v for v in lum if v is not None]
        row = dict(source=src, n=len(vals), A_gray=mu, A_gray_lo=lo, A_gray_hi=hi,
                   lum_smd_mean=float(np.mean(lum)) if lum else np.nan)
        if src in COLOUR_SETS:
            a_col, a_lo, a_hi = ref[src]
            row.update(A_color=a_col, A_color_lo=a_lo, A_color_hi=a_hi)
            den = a_col - 0.5
            r_mid = (mu - 0.5) / den
            r_lo = (lo - 0.5) / den
            r_hi = (hi - 0.5) / den
            row.update(R=r_mid, R_lo=r_lo, R_hi=r_hi)
            if not gate_model:
                dec = 'không kết luận: mô hình xám không đạt đối chứng (BA hoặc AK)'
            elif a_lo <= 0.5:
                dec = 'không có tín hiệu tham chiếu để so sánh'
            elif r_lo >= 0.8:
                dec = 'không phụ thuộc màu: giữ ≥ 80% tín hiệu khi không có màu'
            elif r_hi <= 0.2:
                dec = 'phụ thuộc màu: mất ≥ 80% tín hiệu khi không có màu'
            else:
                dec = 'không kết luận: giữ một phần tín hiệu hoặc CI rộng'
            row['decision'] = dec
        else:
            row.update(A_color=np.nan, A_color_lo=np.nan, A_color_hi=np.nan, R=np.nan, R_lo=np.nan, R_hi=np.nan,
                       decision='không áp dụng: bộ far đã là thang xám')
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'graycolor_decision.csv'), index=False)
    print(f'Mô hình xám: BA {ba_gray[0]:.3f} (ngưỡng {ba_color - 0.05:.3f}); AK {ak_gray[0]:.3f} (ngưỡng 0,55); '
          f'cổng mô hình: {"đạt" if gate_model else "KHÔNG đạt"}')
    pd.set_option('display.width', 280)
    pd.set_option('display.max_columns', 30)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
