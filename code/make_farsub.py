# -*- coding: utf-8 -*-
"""
Tạo tập far-OOD con cho đợt huấn luyện xám (docs/dose_response_design.md mục 12).

Giữ nguyên toàn bộ test_images của retinamnist, pneumoniamnist, breastmnist (224 px).
Lấy ngẫu nhiên 800 ảnh test của bloodmnist và pathmnist (seed 0, chọn bằng numpy.random.default_rng(0)).
Ghi mỗi tệp dưới dạng .npz với khóa test_images, cùng tên tệp như bộ gốc, và in MD5 để truy vết.

Đầu vào: các tệp gốc (đường dẫn trong SRC bên dưới). Đầu ra: thư mục OUT_DIR.
"""
import hashlib
import os

import numpy as np

OUT_DIR = r'C:\data\bcn20000\farsub'
SRC = {
    'retinamnist_224.npz': (r'D:\NCKH_3\ma21_ood_calibration_bcn20000\far_data\retinamnist_224.npz', None),
    'pneumoniamnist_224.npz': (r'D:\NCKH_3\ma21_ood_calibration_bcn20000\far_data\pneumoniamnist_224.npz', None),
    'breastmnist_224.npz': (r'D:\NCKH_3\ma21_ood_calibration_bcn20000\far_data\breastmnist_224.npz', None),
    'bloodmnist_224.npz': (r'C:\data\bcn20000\far_color\bloodmnist_224.npz', 800),
    'pathmnist_224.npz': (r'C:\data\bcn20000\far_color\pathmnist_224.npz', 800),
}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, (src, k) in SRC.items():
        arr = np.load(src)['test_images']
        n0 = len(arr)
        if k is not None and n0 > k:
            idx = np.sort(rng.choice(n0, k, replace=False))
            arr = arr[idx]
        dst = os.path.join(OUT_DIR, name)
        np.savez(dst, test_images=arr)
        print(f'{name}: {n0} -> {len(arr)} ảnh, shape {arr.shape}, md5 {md5(dst)}', flush=True)


if __name__ == '__main__':
    main()
