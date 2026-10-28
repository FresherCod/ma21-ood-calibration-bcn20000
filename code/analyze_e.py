# -*- coding: utf-8 -*-
"""
Phân tích mục 16 (E1, E2, E3, E5) theo đúng quy tắc đã đăng ký (docs/dose_response_design.md, mục 16, 2026-10-21).

Dùng:
  python code/analyze_e.py res   --b DIR_B --r DIR_BR --out analysis/e1_res
  python code/analyze_e.py rot   --b DIR_B --j DIR_J --p DIR_P --out analysis/e2_rot
  python code/analyze_e.py pad   --j DIR_PJ --b DIR_PB --out analysis/e3_pad
  python code/analyze_e.py seeds --j DIR_J --b DIR_B --j2 DIR_J2 --b2 DIR_B2 --out analysis/e5_seeds
Thêm --smoke để chạy với ít seed (không kết luận). Mỗi DIR_* là thư mục outputs (results_seed*.json, far_scores_seed*.csv).

Quy tắc (mục 16, cùng quy ước với mục 14):
- CI: t hai phía 97,5% cho mọi bộ far (jit.t_ci với level 0,9875).
- R: (A − 0,5) / (A_đối chứng − 0,5), mẫu số là điểm ước lượng của đối chứng, cận của tử số lấy từ cận của mẫu (như mục 12).
- E1: cổng C0 (tín hiệu của B, đối chứng), G1, G2 của B_R so với B. G3 không áp dụng. Quyết định theo hiệu A(B_R) − A(B) theo seed.
- E2: điều kiện hợp lệ: BA(P) − BA(B) ≥ −0,03 và |BA(P) − BA(J)| ≤ 0,03 (khớp mức giảm BA của J, mục 15). Cổng C0 của B. Quyết định theo d_J − d_P.
- E3: như mục 14 (cổng và quy tắc R) trên dữ liệu PAD, không K1.
- E5: như mục 14 trên 20 seed, kèm 10 seed để so độ rộng CI.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import analyze_jit as jit
import ood_calibration as oc  # chỉ dùng augment_rotation và AUG_ROT; không huấn luyện

PRIMARY, REPORT_ONLY = jit.PRIMARY, jit.REPORT_ONLY
SRC_ALL = PRIMARY + REPORT_ONLY
CI_LEVEL = 0.9875                          # CI t hai phía 97,5%


def need(a, *names):
    missing = [n for n in names if getattr(a, n) is None]
    if missing:
        raise SystemExit('thiếu tham số: ' + ', '.join('--' + n for n in missing))


def need_seeds(R, name, smoke):
    got = sorted(r['seed'] for r in R)
    if not smoke and got != list(range(10)):
        raise SystemExit(f'{name}: cần đúng seed 0 đến 9, có {got}. Dùng --smoke chỉ để thử.')


def auc_map(R, src):
    return {r['seed']: r['far_ood'][src]['energy_ts']['auroc'] for r in R if src in r.get('far_ood', {})}


def auc_list(R, src):
    return list(auc_map(R, src).values())


def diff_map(RA, RB, src):
    """A(A) − A(B) theo seed, chỉ các seed có ở cả hai nhánh."""
    ma, mb = auc_map(RA, src), auc_map(RB, src)
    return {s: ma[s] - mb[s] for s in sorted(set(ma) & set(mb))}


def c0_ok(R, src):
    """Tín hiệu gốc: cận dưới của CI t hai phía 97,5% của AUROC > 0,6 (mục 14)."""
    v = auc_list(R, src)
    return len(v) >= 2 and jit.t_ci(v, CI_LEVEL)[1] > 0.6


def show_gates(RJ, RB, g, skip_g3=False):
    diffs = g['ba_diff']
    print(f'nhánh J: n={len(RJ)}, BA {np.mean(g["ba_j"]):.4f}, AK {np.mean(g["ak_j"]):.4f}')
    print(f'nhánh B: n={len(RB)}, BA {np.mean(g["ba_b"]):.4f}, AK {np.mean(g["ak_b"]):.4f}')
    print(f'G1 (BA hiệu trung bình {np.mean(diffs):+.4f}, nhỏ nhất {np.min(diffs):+.4f}): {"đạt" if g["g1"] else "KHÔNG đạt"}')
    print(f'G2 (AK_J {np.mean(g["ak_j"]):.4f} ≥ AK_B − 0,05 = {np.mean(g["ak_b"]) - 0.05:.4f}): {"đạt" if g["g2"] else "KHÔNG đạt"}')
    if not skip_g3:
        g3_txt = 'thiếu D1' if g['g3'] is None else ('đạt' if g['g3'] else 'KHÔNG đạt')
        print(f'G3 (S = D1_J / D1_B = {g["s_ratio"]:.4f} ≤ 0,5): {g3_txt}')


def show_table(df):
    pd.set_option('display.width', 240)
    pd.set_option('display.max_colwidth', 90)
    print(df.round(3).to_string(index=False))


# ---------- E1: kiểm soát độ phân giải ----------

def run_res(a):
    need(a, 'b', 'r')
    RB, RR = jit.load(a.b), jit.load(a.r)
    need_seeds(RB, 'B', a.smoke)
    need_seeds(RR, 'B_R', a.smoke)
    g = jit.gates(RR, RB)                  # G1, G2 của B_R so với B; G3 không áp dụng cho E1
    print('E1: B_R so với B (G3 không áp dụng)')
    show_gates(RR, RB, g, skip_g3=True)
    rows = []
    for src in SRC_ALL:
        d = diff_map(RR, RB, src)
        vals = np.array(list(d.values()))
        if len(vals) < 2:
            rows.append(dict(source=src, note='thiếu dữ liệu'))
            continue
        m, lo, hi = jit.t_ci(vals, CI_LEVEL)
        mb, lb, hb = jit.t_ci(auc_list(RB, src), CI_LEVEL)
        mbr, lbr, hbr = jit.t_ci(auc_list(RR, src), CI_LEVEL)
        # Cách A (chính, mục 12): mẫu số là điểm ước lượng của B. Cách B (độ nhạy): cả hai cận cùng đổi.
        r_mid = (mbr - 0.5) / (mb - 0.5)
        r_lo = (lbr - 0.5) / (mb - 0.5)
        r_hi = (hbr - 0.5) / (mb - 0.5)
        if src in PRIMARY:
            # Cực trị trên hình hộp (cận A_BR) × (cận A_B): cùng cách với analyze_jit.decision_rows.
            corners = [(x - 0.5) / (y - 0.5) for x in (lbr, hbr) for y in (lb, hb)]
            r_lo_both, r_hi_both = min(corners), max(corners)
        else:
            r_lo_both = r_hi_both = np.nan
        if src in PRIMARY:
            failed = [k for k, ok in [('C0', c0_ok(RB, src)), ('G1', g['g1']), ('G2', g['g2'])] if not ok]
            if not a.smoke and len(vals) < 10:
                dec = f'không kết luận: thiếu seed ({len(vals)}/10)'
            elif failed:
                dec = 'không kết luận: không đạt ' + ', '.join(failed)
            elif abs(m) >= 0.05 and (lo > 0 or hi < 0):
                dec = 'độ phân giải là nguồn nhiễu đáng kể: kèm cảnh báo mục 14'
            else:
                dec = 'độ phân giải không đổi kết luận ở mức này'
        else:
            dec = 'chỉ báo cáo'
        rows.append(dict(source=src, n=len(vals), diff_mean=m, diff_lo=lo, diff_hi=hi,
                         A_B=mb, A_BR=mbr, R_BR=r_mid, R_BR_lo=r_lo, R_BR_hi=r_hi,
                         R_BR_lo_both_bounds=r_lo_both, R_BR_hi_both_bounds=r_hi_both, decision=dec))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'decision.csv'), index=False)
    show_table(df)


# ---------- E2: placebo hình học ----------

def sharp(x):
    """Trung bình |gradient| theo ngang và dọc của độ sáng, trên tensor NCHW trong [0, 1] (đơn vị: thang [0, 1] mỗi pixel)."""
    g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    dx = (g[:, :, 1:] - g[:, :, :-1]).abs().mean()
    dy = (g[:, 1:, :] - g[:, :-1, :]).abs().mean()
    return float(0.5 * (dx + dy))


def sharpness_placebo(n=256, seed=0):
    """Mục 16 (E2): độ sắc nét của ảnh ID train trước và sau xoay ±30 độ. Xoay trên khung 256 rồi cắt giữa 224,
    đúng thứ tự khi huấn luyện (gpu_prep: xoay trước, cắt sau)."""
    raw = np.memmap(jit.X_PATH, dtype=np.uint8, mode='r')
    rows_n = raw.size // (jit.RESIZE * jit.RESIZE * 3)
    X = raw.reshape(rows_n, jit.RESIZE, jit.RESIZE, 3)
    idx = np.sort(np.random.default_rng(seed).choice(rows_n, size=min(n, rows_n), replace=False))
    frame = torch.from_numpy(np.array(X[idx])).permute(0, 3, 1, 2).float() / 255.0    # (n, 3, 256, 256)
    torch.manual_seed(seed)
    oc.AUG_ROT = 30.0
    rot = oc.augment_rotation(frame, torch.device('cpu'))
    o, c = jit.OFF, jit.CROP
    return sharp(frame[:, :, o:o + c, o:o + c]), sharp(rot[:, :, o:o + c, o:o + c])


def run_rot(a):
    need(a, 'b', 'j', 'p')
    RB, RJ, RP = jit.load(a.b), jit.load(a.j), jit.load(a.p)
    for R, name in [(RB, 'B'), (RJ, 'J'), (RP, 'P')]:
        need_seeds(R, name, a.smoke)
    ba_pb = jit.gates(RP, RB)['ba_diff']   # BA(P) − BA(B), theo seed
    ba_pj = jit.gates(RP, RJ)['ba_diff']   # BA(P) − BA(J), theo seed
    valid_b = bool(np.mean(ba_pb) >= -0.03)
    valid_j = bool(abs(np.mean(ba_pj)) <= 0.03)
    print(f'E2: BA(P) − BA(B) = {np.mean(ba_pb):+.4f} (cần ≥ −0,03): {"đạt" if valid_b else "KHÔNG đạt"}')
    print(f'E2: BA(P) − BA(J) = {np.mean(ba_pj):+.4f} (cần |·| ≤ 0,03, khớp mức giảm BA của J theo mục 15): {"đạt" if valid_j else "KHÔNG đạt"}')
    rows = []
    for src in SRC_ALL:
        dj = diff_map(RB, RJ, src)        # d_J = A(B) − A(J)
        dp = diff_map(RB, RP, src)        # d_P = A(B) − A(P)
        seeds = sorted(set(dj) & set(dp))
        if len(seeds) < 2:
            rows.append(dict(source=src, note='thiếu dữ liệu'))
            continue
        j = np.array([dj[s] for s in seeds])
        p = np.array([dp[s] for s in seeds])
        mj = jit.t_ci(j, CI_LEVEL)[0]
        mp = jit.t_ci(p, CI_LEVEL)[0]
        m, lo, hi = jit.t_ci(j - p, CI_LEVEL)    # d_J − d_P = A(P) − A(J) theo seed
        if src not in PRIMARY:
            dec = 'chỉ báo cáo'
        elif not valid_b:
            dec = 'không kết luận: P không hợp lệ, BA(P) − BA(B) < −0,03'
        elif not valid_j:
            dec = 'không kết luận: P không khớp mức giảm BA của J'
        elif not c0_ok(RB, src):
            dec = 'không kết luận: không đạt C0 (tín hiệu gốc của B)'
        elif m >= 0.05 and lo > 0:
            dec = 'tác động của J đặc hiệu hơn tác động hình học'
        elif abs(m) < 0.05:
            dec = 'tác động của J không đặc hiệu (|d_J − d_P| < 0,05)'
        else:
            dec = 'ngoài hai nhánh đã đăng ký: báo cáo nguyên văn'
        rows.append(dict(source=src, n=len(seeds), d_J=mj, d_P=mp,
                         dJ_minus_dP=m, lo=lo, hi=hi, decision=dec))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'decision.csv'), index=False)
    show_table(df)
    s_orig, s_rot = sharpness_placebo()
    pd.DataFrame([dict(sharp_orig=s_orig, sharp_rot=s_rot, ratio_rot_over_orig=s_rot / s_orig)]).to_csv(
        os.path.join(a.out, 'sharpness.csv'), index=False)
    print(f'độ sắc nét (256 ảnh ID train, cắt 224, thang [0,1]): gốc {s_orig:.4f}, xoay ±30° {s_rot:.4f}, tỉ lệ {s_rot / s_orig:.3f}')


# ---------- E3: tập ID thứ hai (PAD-UFES-20) ----------

def run_pad(a):
    need(a, 'j', 'b')
    RJ, RB = jit.load(a.j), jit.load(a.b)
    need_seeds(RJ, 'J_pad', a.smoke)
    need_seeds(RB, 'B_pad', a.smoke)
    g = jit.gates(RJ, RB)
    print('E3: J_pad so với B_pad (PAD-UFES-20)')
    show_gates(RJ, RB, g)
    df = pd.DataFrame(jit.decision_rows(RJ, RB, g, SRC_ALL))
    df.to_csv(os.path.join(a.out, 'decision.csv'), index=False)
    show_table(df)


# ---------- E5: thêm 10 seed ----------

def run_seeds(a):
    need(a, 'j', 'b', 'j2', 'b2')
    RJ10, RB10 = jit.load(a.j), jit.load(a.b)
    RJ2, RB2 = jit.load(a.j2), jit.load(a.b2)
    need_seeds(RJ10, 'J (10 seed đầu)', a.smoke)
    need_seeds(RB10, 'B (10 seed đầu)', a.smoke)
    RJ20, RB20 = RJ10 + RJ2, RB10 + RB2
    if not a.smoke:
        for R, name in [(RJ20, 'J'), (RB20, 'B')]:
            if sorted(r['seed'] for r in R) != list(range(20)):
                raise SystemExit(f'{name}: 20 seed phải là 0 đến 19, không trùng')
    for n, RJ_, RB_ in [(10, RJ10, RB10), (20, RJ20, RB20)]:
        g = jit.gates(RJ_, RB_)
        print(f'\nE5, {n} seed:')
        show_gates(RJ_, RB_, g)
        df = pd.DataFrame(jit.decision_rows(RJ_, RB_, g, SRC_ALL))
        df.insert(0, 'n_seed', n)
        df.to_csv(os.path.join(a.out, f'decision_{n}seed.csv'), index=False)
        show_table(df)
        if 'R_lo' in df.columns:
            width = {r.source: round(r.R_hi - r.R_lo, 3) for r in df.itertuples() if not pd.isna(r.R_lo)}
            print('độ rộng CI của R (cận trên − cận dưới):', width)


def main():
    ap = argparse.ArgumentParser(description='Phân tích mục 16: E1 res, E2 rot, E3 pad, E5 seeds')
    ap.add_argument('exp', choices=['res', 'rot', 'pad', 'seeds'])
    ap.add_argument('--b')
    ap.add_argument('--r')
    ap.add_argument('--j')
    ap.add_argument('--p')
    ap.add_argument('--j2')
    ap.add_argument('--b2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {'res': run_res, 'rot': run_rot, 'pad': run_pad, 'seeds': run_seeds}[a.exp](a)


if __name__ == '__main__':
    main()
