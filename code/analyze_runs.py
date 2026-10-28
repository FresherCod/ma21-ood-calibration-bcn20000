# -*- coding: utf-8 -*-
"""
Phân tích nhiều lần chạy. Mỗi lần chạy là một thư mục OUT chứa results_seed*.json.

Xuất ba bảng (CSV) vào thư mục --out và in ra màn hình:
1. summary.csv: trung bình, độ lệch chuẩn, khoảng tin cậy t 95% theo seed, min, max.
2. contrasts.csv: đối chiếu ghép cặp theo seed trong cùng một lần chạy
   (far-OOD trừ đối chứng xám, màu trừ thang xám, đặc trưng trừ logit).
3. cross_run.csv: so sánh lần chạy đầu tiên với các lần chạy còn lại trên các seed chung.

Ví dụ:
    python code/analyze_runs.py --out analysis_v4 kaggle_output/ma21_bcn20000/v3/outputs
    python code/analyze_runs.py --out analysis_cmp v3_dir v4_dir
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ['msp_ts', 'energy_ts', 'mc_msp', 'maha', 'knn']          # điểm OOD chính để đối chiếu
KEY_ID = ['ts_ece', 'ts_nll', 'ts_acc', 'ts_bal_acc', 'raw_ece', 'raw_nll']
KEY_CROSS = KEY_ID + ['auroc_energy_ts', 'auroc_maha', 'auroc_knn']


def load_run(out_dir):
    """Đọc mọi results_seed*.json trong một thư mục OUT thành một bảng, một dòng mỗi seed."""
    rows, config = [], None
    for p in sorted(glob.glob(os.path.join(out_dir, 'results_seed*.json'))):
        with open(p, encoding='utf-8') as f:
            r = json.load(f)
        config = config or r.get('config', {})
        row = {'seed': r['seed'], 'best_epoch': r['best_epoch'], 'T': r['temperature'],
               'n_epochs_run': len(r['history'])}
        for k, v in r['id_metrics_ts'].items():
            row[f'ts_{k}'] = v
        for k, v in r['id_metrics_raw'].items():
            row[f'raw_{k}'] = v
        for k, v in r['ood'].items():
            row[f'auroc_{k}'] = v['auroc']
            row[f'fpr95_{k}'] = v['fpr95']
        for src, d in r.get('far_ood', {}).items():
            for k, v in d.items():
                if isinstance(v, dict):
                    row[f'far_{src}_auroc_{k}'] = v['auroc']
                    row[f'far_{src}_fpr95_{k}'] = v['fpr95']
        for src, d in r.get('far_ood_gray', {}).items():
            for k, v in d.items():
                if isinstance(v, dict):
                    row[f'farg_{src}_auroc_{k}'] = v['auroc']
                    row[f'farg_{src}_fpr95_{k}'] = v['fpr95']
        for k, v in r.get('gray_control', {}).items():
            if isinstance(v, dict):
                row[f'gray_{k}_auroc'] = v['auroc']
                row[f'gray_{k}_fpr95'] = v['fpr95']
        for src, d in r.get('lowlevel', {}).items():
            for k, v in d.items():
                row[f'll_{src}_auroc_{k}'] = v['auroc']
                row[f'll_{src}_fpr95_{k}'] = v['fpr95']
        for src, d in r.get('matched', {}).items():
            for k, v in d.items():
                if v is not None:
                    row[f'mt_{src}_{k}'] = v
        rows.append(row)
    if not rows:
        raise SystemExit(f'Không có results_seed*.json trong {out_dir}')
    df = pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)
    df.attrs['config'] = config or {}
    return df


def t_ci(x):
    """Trung bình, độ lệch chuẩn và khoảng tin cậy t 95% (n nhỏ nên dùng phân phối t)."""
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, np.nan
    sd = float(x.std(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - h, m + h


def paired(d):
    """Hiệu ghép cặp theo seed: trung bình, khoảng tin cậy t 95%, p hai phía (kiểm định t một mẫu)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    m, _, lo, hi = t_ci(d)
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 else np.nan
    return m, lo, hi, p, len(d)


def summary_table(name, df):
    rows = []
    for col in df.columns:
        if col in ('seed',) or not np.issubdtype(df[col].dtype, np.number):
            continue
        x = df[col].dropna()
        if len(x) == 0:
            continue
        m, sd, lo, hi = t_ci(x)
        rows.append(dict(run=name, metric=col, n=len(x), mean=m, sd=sd, ci_lo=lo, ci_hi=hi,
                         min=float(x.min()), max=float(x.max())))
    return pd.DataFrame(rows)


def contrasts_table(name, df):
    """Các đối chiếu ghép cặp trong một lần chạy. Điểm dương nghĩa là vế đầu tiên phát hiện OOD tốt hơn."""
    srcs = sorted({c.split('_auroc_')[0][len('far_'):] for c in df.columns
                   if c.startswith('far_') and '_auroc_' in c})
    rows = []

    def add(label, a, b):
        if a in df and b in df:
            m, lo, hi, p, n = paired(df[a] - df[b])
            rows.append(dict(run=name, contrast=label, n=n, diff=m, ci_lo=lo, ci_hi=hi, p=p))

    for k in METHODS:
        for src in srcs:
            add(f'far {src} − xám [{k}]', f'far_{src}_auroc_{k}', f'gray_{k}_auroc')
            add(f'far {src} màu − thang xám [{k}]', f'far_{src}_auroc_{k}', f'farg_{src}_auroc_{k}')
    for src in srcs:
        add(f'far {src}: maha − energy_ts', f'far_{src}_auroc_maha', f'far_{src}_auroc_energy_ts')
        add(f'far {src}: knn − energy_ts', f'far_{src}_auroc_knn', f'far_{src}_auroc_energy_ts')
        for k in ['energy_ts', 'maha', 'knn']:
            add(f'far {src}: {k} giảm khi khớp mức thấp', f'far_{src}_auroc_{k}', f'mt_{src}_{k}')
        for lk in ['sat_high_is_id', 'grad_low_is_id']:
            add(f'far {src}: maha − mức thấp [{lk}]', f'far_{src}_auroc_maha', f'll_{src}_auroc_{lk}')
            add(f'far {src}: energy_ts − mức thấp [{lk}]', f'far_{src}_auroc_energy_ts', f'll_{src}_auroc_{lk}')
    add('OOD lớp giữ lại: maha − energy_ts', 'auroc_maha', 'auroc_energy_ts')
    add('OOD lớp giữ lại: knn − energy_ts', 'auroc_knn', 'auroc_energy_ts')
    add('xám (ID): energy_ts − msp_ts', 'gray_energy_ts_auroc', 'gray_msp_ts_auroc')
    return pd.DataFrame(rows)


def cross_run_table(runs):
    """So sánh từng lần chạy với lần đầu tiên, ghép theo seed chung."""
    names = list(runs)
    ref_name = names[0]
    ref = runs[ref_name]
    rows = []
    for other_name in names[1:]:
        oth = runs[other_name]
        common = sorted(set(ref['seed']) & set(oth['seed']))
        if not common:
            continue
        a = ref.set_index('seed').loc[common]
        b = oth.set_index('seed').loc[common]
        for col in KEY_CROSS:
            if col in a and col in b:
                m, lo, hi, p, n = paired(b[col] - a[col])
                rows.append(dict(reference=ref_name, other=other_name, seeds=','.join(map(str, common)),
                                 metric=col, n=n, diff_other_minus_ref=m, ci_lo=lo, ci_hi=hi, p=p))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Phân tích và so sánh các lần chạy mã 21')
    ap.add_argument('dirs', nargs='+', help='thư mục OUT chứa results_seed*.json')
    ap.add_argument('--out', default='analysis', help='thư mục ghi bảng CSV')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {}
    for d in args.dirs:
        name = os.path.normpath(d).replace(os.sep, '/').split('/')[-2 if d.rstrip('/\\').endswith('outputs') else -1]
        df = load_run(d)
        base, k = name, 2
        while name in runs:                          # tránh trùng tên khi hai thư mục cùng tên
            name = f'{base}#{k}'
            k += 1
        runs[name] = df
        cfg = df.attrs.get('config', {})
        print(f'== {name}: {len(df)} seed {df["seed"].tolist()} | cấu hình: '
              f'BACKBONE={cfg.get("BACKBONE", "effb0")} EPOCHS={cfg.get("EPOCHS")} PATIENCE={cfg.get("PATIENCE")} '
              f'HOLDOUT={cfg.get("HOLDOUT")}')

    summ = pd.concat([summary_table(n, df) for n, df in runs.items()], ignore_index=True)
    cont = pd.concat([contrasts_table(n, df) for n, df in runs.items()], ignore_index=True)
    cross = cross_run_table(runs) if len(runs) > 1 else pd.DataFrame()

    summ.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    cont.to_csv(os.path.join(args.out, 'contrasts.csv'), index=False)
    if len(cross):
        cross.to_csv(os.path.join(args.out, 'cross_run.csv'), index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 500)
    print('\n== Đối chiếu ghép cặp theo seed (diff, CI 95%, p):')
    print(cont.round(4).to_string(index=False))
    if len(cross):
        print('\n== So với lần đầu tiên, trên seed chung:')
        print(cross.round(4).to_string(index=False))
    print(f'\nĐã ghi vào {args.out}/: summary.csv, contrasts.csv' + (', cross_run.csv' if len(cross) else ''))


if __name__ == '__main__':
    main()
