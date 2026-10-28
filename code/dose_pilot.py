# -*- coding: utf-8 -*-
"""
Pilot khả thi của thiết kế liều v0.3 (docs/dose_response_design.md mục 9): chỉ dùng thống kê ảnh, không chấm mô hình.
Mirror đúng thứ tự gọi RNG của imagelevel_dose.run_seed để θ và mẫu far trùng với lần chạy thật.
Báo cáo theo (seed, bộ, liều): θ, cos, chênh lệch độ lớn chroma, SMD độ bão hòa và L của dir và rot, tỉ số ΔE76 rot/dir.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_calibration as m  # noqa: E402
import imagelevel_control as ic  # noqa: E402
import imagelevel_chroma as ch  # noqa: E402
import imagelevel_dose as dz  # noqa: E402

OUT_CSV = os.environ.get('PILOT_OUT', r'C:\data\bcn20000\dose_pilot.csv')


def main():
    df = pd.read_csv(m.DATA_CSV)
    df = df[df['split'].astype(str).str.lower() == 'train'].reset_index(drop=True)
    names = df['bcn_filename'].tolist()
    rowmap = {f: i for i, f in enumerate(names)}
    X = np.memmap(ic.IMG_MEMMAP, dtype=np.uint8, mode='r', shape=(len(names), m.RESIZE, m.RESIZE, 3))
    Xfar, far_src = m.build_far_array()
    rows = []
    for seed in [int(s) for s in m.SEEDS]:
        S = m.build_split(seed, rowmap)
        row_id = S['idd']['row'].values
        rng = np.random.default_rng(seed + 100)
        te = S['te']
        te_sub = rng.choice(te, min(ic.N_ID_SUB, len(te)), replace=False)
        id_imgs = np.asarray(X[row_id[te_sub]])
        lab_id = ch.to_lab(id_imgs)
        st_id = ch.lab_stats(lab_id)
        n = len(id_imgs)
        sat_id, _ = m.lowlevel_scores(id_imgs, np.arange(n))
        for src in np.unique(far_src):
            mk = np.flatnonzero(far_src == src)
            pick = rng.choice(mk, min(ic.N_FAR_SUB, len(mk)), replace=False)
            far_imgs = np.asarray(Xfar[pick])
            nf = len(far_imgs)
            gray = ic.is_gray_set(far_imgs)
            lab_far = ch.to_lab(far_imgs)
            st_far = ch.lab_stats(lab_far)
            sat_far, grad_far = m.lowlevel_scores(far_imgs, np.arange(nf))
            lum_far = dz.lum_of(far_imgs)
            if gray:
                for alpha in dz.DOSES:
                    lab_dir = ch.chroma_match(lab_id, dz.dir_targets(st_id, st_far, alpha))
                    dir_img = ch.from_lab(lab_dir)
                    sat_d, _ = m.lowlevel_scores(dir_img, np.arange(n))
                    rows.append(dict(seed=seed, source=src, gray=True, dose=alpha, feasible=None, theta=None,
                                     cos=None, mag_rel=None, smd_sat_dir=ic.smd(sat_d, sat_far),
                                     smd_lum_dir=ic.smd(dz.lum_of(dir_img), lum_far), smd_sat_rot=None,
                                     smd_lum_rot=None, de_ratio=None))
                continue
            for alpha in dz.DOSES:
                lab_dir = ch.chroma_match(lab_id, dz.dir_targets(st_id, st_far, alpha))
                dir_img = ch.from_lab(lab_dir)
                pb = dz.placebo_theta(lab_id, st_id, st_far, alpha, rng)
                rec = dict(seed=seed, source=src, gray=False, dose=alpha,
                           feasible=None if pb is None else pb['feasible'],
                           theta=None if pb is None else pb.get('theta_deg'),
                           cos=None if pb is None else pb.get('cos'),
                           mag_rel=None if pb is None else pb.get('mag_rel'))
                sat_d, _ = m.lowlevel_scores(dir_img, np.arange(n))
                rec['smd_sat_dir'] = ic.smd(sat_d, sat_far)
                rec['smd_lum_dir'] = ic.smd(dz.lum_of(dir_img), lum_far)
                de_dir = dz.delta_e76(lab_dir, lab_id)
                if pb is not None and pb['feasible']:
                    lab_rot = ch.chroma_match(lab_id, dz.rot_targets(st_id, st_far, alpha, pb['theta_deg']))
                    rot_img = ch.from_lab(lab_rot)
                    sat_r, _ = m.lowlevel_scores(rot_img, np.arange(n))
                    rec['smd_sat_rot'] = ic.smd(sat_r, sat_far)
                    rec['smd_lum_rot'] = ic.smd(dz.lum_of(rot_img), lum_far)
                    rec['de_ratio'] = dz.delta_e76(lab_rot, lab_id) / max(de_dir, 1e-9)
                else:
                    rec['smd_sat_rot'] = rec['smd_lum_rot'] = rec['de_ratio'] = None
                rows.append(rec)
            print(f'seed {seed} {src} xong', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(out.round(3).to_string(index=False))
    # tóm tắt theo bộ và liều, trung bình 5 seed
    colour = out[~out['gray']]
    summ = colour.groupby(['source', 'dose']).agg(
        feasible_all=('feasible', lambda x: bool(all(v is True for v in x))),
        theta_mean=('theta', 'mean'), cos_max=('cos', lambda x: float(np.max(np.abs(x.dropna()))) if x.notna().any() else np.nan),
        mag_rel_max=('mag_rel', 'max'),
        sat_dir=('smd_sat_dir', 'mean'), sat_rot=('smd_sat_rot', 'mean'),
        lum_dir=('smd_lum_dir', 'mean'), lum_rot=('smd_lum_rot', 'mean'),
        de_ratio_mean=('de_ratio', 'mean'))
    print()
    print(summ.round(3).to_string())
    summ.to_csv(OUT_CSV.replace('.csv', '_summary.csv'))


if __name__ == '__main__':
    main()
