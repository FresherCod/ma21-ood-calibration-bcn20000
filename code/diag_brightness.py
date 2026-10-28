# -*- coding: utf-8 -*-
"""
Chẩn đoán đơn biến: độ sáng, tương phản và gradient có tách được ID test khỏi far-OOD không.

Không huấn luyện, không dùng GPU. Đọc checkpoint đã có ở đây chỉ để lấy điểm của mô hình (AUROC năng lượng, đã lưu).
- ID test của từng seed: lấy từ predictions_seed*.csv (is_ood = 0), ảnh đọc từ X_train_256.u8.
- Far: tệp far con trong farsub (224 px). Đưa về đúng khung mô hình nhìn thấy: phóng 256 bilinear rồi cắt giữa 224
  (far_model_frame). Khung này khác khung 224 gốc: lệch AUROC độ sáng đáng kể (xem đính chính trong LOG.md 2026-10-20).
- Ảnh được chuyển sang luminance. ID được cắt giữa 224 px từ X_train_256 (cùng offset với mô hình).
- AUROC đơn biến: ID là lớp dương, điểm càng cao càng giống ID (như AUROC của mô hình).
  AUROC gần 0,5 là không tách được. AUROC_abs = max(a, 1 − a) bỏ qua chiều.
- SMD độ sáng = (mean ID − mean far) / sd far, cùng công thức với far_lum_smd trong code.

Chạy: python code/diag_brightness.py
"""
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = 'D:/NCKH_3/ma21_ood_calibration_bcn20000'
CSV = f'{ROOT}/data/bcn_20k_train.csv'
X_PATH = 'C:/data/bcn20000/X_train_256.u8'
ZIP_PATH = 'C:/data/bcn20000/BCN_20k_train.zip'
FAR_DIR = 'C:/data/bcn20000/farsub'
RES_DIR = f'{ROOT}/kaggle_output/ma21_bcn20000/v5g/outputs'
OUT = f'{ROOT}/analysis/diag_brightness'
SEEDS = list(range(10))
RESIZE, CROP = 256, 224
FAR_SETS = ['retinamnist_224', 'pneumoniamnist_224', 'breastmnist_224', 'bloodmnist_224', 'pathmnist_224']


def luminance(a):
    """Ảnh uint8 (n, H, W, 3) -> luminance float32 (n, H, W), cùng trọng số với gray3_np.
    Ảnh đã là thang xám (n, H, W) (breastmnist, pneumoniamnist) thì giữ nguyên."""
    a = a.astype(np.float32)
    if a.ndim == 3:
        return a
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def far_model_frame(arr):
    """Đưa far về đúng khung mô hình nhìn thấy: ảnh 224 px (màu hoặc xám) -> RGB -> phóng 256 bilinear -> cắt giữa 224.
    Khớp với build_far_array và với crop đánh giá (offset (RESIZE - CROP) // 2). Trả về (n, CROP, CROP, 3) uint8."""
    off = (RESIZE - CROP) // 2
    out = np.empty((len(arr), CROP, CROP, 3), dtype=np.uint8)
    for i, a in enumerate(arr):
        rgb = np.stack([a] * 3, axis=-1) if a.ndim == 2 else a.astype(np.uint8)
        im = Image.fromarray(rgb).convert('RGB').resize((RESIZE, RESIZE), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.uint8)[off:off + CROP, off:off + CROP]
    return out


def features(L):
    """Ba đặc trưng đơn biến cho mỗi ảnh: độ sáng trung bình, độ lệch chuẩn (tương phản), gradient trung bình."""
    gx = np.abs(np.diff(L, axis=2))[:, :-1, :]
    gy = np.abs(np.diff(L, axis=1))[:, :, :-1]
    return {
        'brightness': L.mean(axis=(1, 2)),
        'contrast': L.std(axis=(1, 2)),
        'gradient': (gx + gy).mean(axis=(1, 2)),
    }


def train_names():
    df = pd.read_csv(CSV)
    if 'split' in df.columns:
        df = df[df['split'].astype(str).str.lower() == 'train']
    return df['bcn_filename'].tolist()


def check_cache_matches_zip(X, rowmap):
    """Kiểm tra X_train_256 khớp với ảnh gốc trong zip (cùng cách giải mã và thu nhỏ). Dừng nếu lệch."""
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = {os.path.basename(i.filename): i.filename for i in z.infolist() if not i.is_dir()}
        for name in list(rowmap)[:2]:
            im = Image.open(io.BytesIO(z.read(members[name])))
            im.draft('RGB', (RESIZE * 2, RESIZE * 2))
            arr = np.asarray(im.convert('RGB').resize((RESIZE, RESIZE), Image.BILINEAR), dtype=np.uint8)
            diff = np.abs(arr.astype(int) - X[rowmap[name]].astype(int)).max()
            if diff > 2:
                raise SystemExit(f'X_train_256 không khớp zip (chênh lệch tối đa {diff} cho {name}). Dừng.')
    print('kiểm tra X_train_256 với zip: khớp')


def main():
    names = train_names()
    rowmap = {n: i for i, n in enumerate(names)}
    n_rows = os.path.getsize(X_PATH) // (RESIZE * RESIZE * 3)
    assert n_rows == len(names), f'X có {n_rows} dòng, CSV có {len(names)} ảnh train'
    X = np.memmap(X_PATH, dtype=np.uint8, mode='r', shape=(n_rows, RESIZE, RESIZE, 3))
    check_cache_matches_zip(X, rowmap)

    # Far: cắt giữa không cần, far đã là 224 px
    far_feats, far_n = {}, {}
    for src in FAR_SETS:
        arr = np.load(f'{FAR_DIR}/{src}.npz')['test_images']
        far_feats[src] = features(luminance(far_model_frame(arr)))
        far_n[src] = len(arr)
        print(f'far {src}: {len(arr)} ảnh')

    rows = []
    for seed in SEEDS:
        pred = pd.read_csv(f'{RES_DIR}/predictions_seed{seed}.csv')
        id_names = pred.loc[pred['is_ood'] == 0, 'bcn_filename'].tolist()
        idx = np.sort([rowmap[n] for n in id_names])
        ix = X[idx][:, (RESIZE - CROP) // 2:(RESIZE + CROP) // 2, (RESIZE - CROP) // 2:(RESIZE + CROP) // 2]
        id_f = features(luminance(np.asarray(ix)))
        with open(f'{RES_DIR}/results_seed{seed}.json', encoding='utf-8') as f:
            res = json.load(f)
        for src in FAR_SETS:
            model_auc = res['far_ood'][src]['energy_ts']['auroc']
            for feat in ['brightness', 'contrast', 'gradient']:
                a, b = id_f[feat], far_feats[src][feat]
                y = np.r_[np.ones(len(a)), np.zeros(len(b))]
                s = np.r_[a, b]
                auc = roc_auc_score(y, s)
                smd = (a.mean() - b.mean()) / b.std(ddof=1)
                rows.append(dict(seed=seed, source=src, feature=feat, auroc=auc,
                                 auroc_abs=max(auc, 1 - auc), smd_id_minus_far=smd,
                                 model_energy_auroc=model_auc, n_id=len(a), n_far=len(b)))
        print(f'seed {seed}: xong ({len(id_names)} ID test)')

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f'{OUT}/per_seed.csv', index=False)
    agg = df.groupby(['source', 'feature']).agg(
        auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'),
        auroc_abs_mean=('auroc_abs', 'mean'), smd_mean=('smd_id_minus_far', 'mean'),
        model_energy_auroc=('model_energy_auroc', 'mean')).reset_index()
    agg.to_csv(f'{OUT}/summary.csv', index=False)
    pd.set_option('display.width', 220)
    print(agg.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
