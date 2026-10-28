# -*- coding: utf-8 -*-
"""
Đối chứng ở mức ảnh cho far-OOD.

Với mỗi bộ far-OOD và mỗi seed:
  1. Biến đổi ảnh ID test để khớp phân bố màu và cường độ với bộ far
     (histogram matching theo từng kênh; nếu bộ far là thang xám thì đổi ID sang thang xám rồi khớp độ sáng).
  2. Làm mờ ảnh ID (Gaussian) để độ dốc trung bình khớp với bộ far, nếu đạt được. Không thể làm sắc nét hơn.
  3. Chấm lại ảnh ID đã biến đổi bằng đúng checkpoint, với cùng nhiệt độ T của lần chạy gốc.

Đo bốn AUROC (điểm càng cao càng giống ID):
  auroc_orig:      ID gốc so với far gốc (đối chiếu với kết quả chính)
  auroc_matched:   ID đã biến đổi so với far gốc (đối chứng chính: nếu về ~0,5 thì tín hiệu nằm trong các thống kê đã khớp)
  auroc_shift:     ID gốc so với ID đã biến đổi (tác động của chính phép biến đổi lên điểm số)
  balance:         SMD của độ bão hòa và độ dốc giữa ID đã biến đổi và far (gần 0 là khớp tốt)

Biến môi trường chính: BACKBONE, CKPT_DIR, RESULTS_DIR, OUT, SEEDS, DATA_CSV, IMG_MEMMAP, FAR_NPZ,
N_ID_SUB, N_BANK (0 = toàn bộ train), N_FAR_SUB, PRETRAINED=0, REQUIRE_GPU.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_calibration as m  # noqa: E402  (đọc cấu hình từ biến môi trường)

import torch  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

N_ID_SUB = int(os.environ.get('N_ID_SUB', 600))
N_BANK = int(os.environ.get('N_BANK', 0))             # 0 = toàn bộ train, khớp với kết quả chính
N_FAR_SUB = int(os.environ.get('N_FAR_SUB', 800))
CKPT_DIR = os.environ.get('CKPT_DIR', m.CKPT_DIR)
RESULTS_DIR = os.environ.get('RESULTS_DIR', m.OUT)
OUT = os.environ.get('OUT', os.path.join(m.RUN_ROOT, 'imagelevel'))
IMG_MEMMAP = os.environ.get('IMG_MEMMAP', r'C:\data\bcn20000\X_train_256.u8')
BLUR_GRID = [round(x, 2) for x in np.arange(0.0, 3.01, 0.25)]     # sigma làm mờ (0 = không làm mờ)
SHARP_GRID = [round(x, 2) for x in np.arange(0.25, 2.01, 0.25)]    # hệ số làm sắc nét (unsharp mask)


def hist_lut(src, ref):
    """Bảng tra 256 mức để phân bố src khớp với phân bố ref (theo CDF).
    Mỗi mức nguồn v ↦ mức nhỏ nhất r của tham chiếu sao cho CDF_ref(r) >= CDF_src(v). Không nội suy,
    nên mức trên cùng của nguồn không bị đẩy lên 255 khi tham chiếu tối hơn."""
    hs = np.bincount(src.ravel(), minlength=256).astype(float)
    hr = np.bincount(ref.ravel(), minlength=256).astype(float)
    cs = np.minimum(np.cumsum(hs) / max(hs.sum(), 1.0), 1.0)
    cr = np.cumsum(hr) / max(hr.sum(), 1.0)
    cr[-1] = 1.0
    lut = np.searchsorted(cr, cs, side='left')
    return np.clip(lut, 0, 255).astype(np.uint8)


def is_gray_set(far_imgs):
    """Bộ far là thang xám nếu ba kênh giống nhau ở mọi điểm ảnh (lấy mẫu để nhanh)."""
    s = far_imgs[:64]
    return bool(np.all(s[..., 0] == s[..., 1]) and np.all(s[..., 1] == s[..., 2]))


def color_match(id_imgs, far_imgs):
    """Khớp màu và cường độ của ID với far. Trả về ảnh uint8 cùng kích thước với id_imgs."""
    if is_gray_set(far_imgs):
        lum_id = np.round(0.299 * id_imgs[..., 0] + 0.587 * id_imgs[..., 1] + 0.114 * id_imgs[..., 2]).astype(np.uint8)
        lut = hist_lut(lum_id, far_imgs[..., 0])
        lum_new = lut[lum_id]
        return np.repeat(lum_new[..., None], 3, axis=-1)
    out = np.empty_like(id_imgs)
    for c in range(3):
        lut = hist_lut(id_imgs[..., c], far_imgs[..., c])
        out[..., c] = lut[id_imgs[..., c]]
    return out


def mean_grad(imgs):
    """Độ dốc trung bình theo điểm ảnh (cùng định nghĩa với lowlevel_scores), cho một lô ảnh uint8."""
    _, grad = m.lowlevel_scores(imgs, np.arange(len(imgs)))
    return grad


def _apply_chunked(imgs, fn):
    out = np.empty_like(imgs)
    for i in range(0, len(imgs), 100):
        chunk = imgs[i:i + 100].astype(np.float32)
        out[i:i + 100] = np.clip(fn(chunk), 0, 255).round().astype(np.uint8)
    return out


def transform_to_grad(imgs, target_grad):
    """Chọn phép biến đổi (làm mờ hoặc làm sắc nét) để độ dốc trung bình median gần target_grad nhất.
    Làm mờ khi ID sắc hơn bộ far; làm sắc nét khi ID mờ hơn. Trả về (ảnh, loại, tham số, độ dốc đạt được)."""
    cands = [('none', 0.0, imgs)]
    for s in BLUR_GRID[1:]:
        cands.append(('blur', s, _apply_chunked(imgs, lambda x, s=s: gaussian_filter(x, sigma=(0, s, s, 0)))))
    for a in SHARP_GRID:
        cands.append(('sharpen', a, _apply_chunked(
            imgs, lambda x, a=a: x + a * (x - gaussian_filter(x, sigma=(0, 1.0, 1.0, 0))))))
    best = None
    for kind, p, out in cands:
        g = float(np.median(mean_grad(out)))
        gap = abs(g - target_grad)
        if best is None or gap < best[0]:
            best = (gap, kind, p, g, out)
    _, kind, p, g, out = best
    return out, kind, p, g


def auroc_id_vs_far(s_id, s_far):
    y = np.r_[np.ones(len(s_id)), np.zeros(len(s_far))]
    return float(roc_auc_score(y, np.r_[s_id, s_far]))


def smd(x_a, x_b):
    """SMD của trung bình x_a so với x_b (chuẩn hóa theo độ lệch chuẩn của x_b).
    Khi x_b hằng số (ví dụ bộ xám có độ bão hòa 0): 0 nếu hai trung bình bằng nhau, None nếu không định nghĩa được."""
    diff = float(np.mean(x_a) - np.mean(x_b))
    sd = float(np.std(x_b, ddof=1)) if len(x_b) > 1 else 0.0
    if sd <= 0:
        return 0.0 if abs(diff) < 1e-9 else None
    return diff / sd


def scores_from_feats(feats, logits, T, maha, bank, knn_k):
    s = {}
    s['energy_ts'] = (T * torch.logsumexp(logits / T, 1)).numpy()
    s['msp_ts'] = torch.softmax(logits / T, 1).max(1).values.numpy()
    s['maha'] = m.maha_score(feats, *maha).numpy()
    s['knn'] = m.knn_score(feats, bank, knn_k).numpy()
    return s


def run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp):
    t0 = time.time()
    S = m.build_split(seed, rowmap)
    K = len(S['classes'])
    idd, y = S['idd'], S['y']
    row_id = idd['row'].values
    rng = np.random.default_rng(seed + 100)

    res_path = os.path.join(RESULTS_DIR, f'results_seed{seed}.json')
    with open(res_path, encoding='utf-8') as f:
        T = float(json.load(f)['temperature'])

    model = m.build_model(K).to(device)
    ck = torch.load(os.path.join(CKPT_DIR, f'seed{seed}', 'best_model.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state'])
    model.eval()

    tr = S['tr']
    bank_pick = np.arange(len(tr)) if N_BANK <= 0 else rng.choice(len(tr), min(N_BANK, len(tr)), replace=False)
    bank_rows = row_id[tr][bank_pick]
    f_bank, _, _ = m.get_feats_logits(model, X, bank_rows, y[tr][bank_pick], device, use_amp)
    maha = m.fit_maha(f_bank, torch.as_tensor(y[tr][bank_pick]), K)
    bank = f_bank

    te = S['te']
    te_sub = rng.choice(te, min(N_ID_SUB, len(te)), replace=False)
    id_rows = row_id[te_sub]
    id_imgs = np.asarray(X[id_rows])                    # ảnh gốc, đúng thứ tự te_sub
    f_id, L_id, _ = m.get_feats_logits(model, id_imgs, np.arange(len(id_imgs)), np.full(len(id_imgs), -1),
                                       device, use_amp)
    s_id = scores_from_feats(f_id, L_id, T, maha, bank, m.KNN_K)
    sat_id, grad_id = m.lowlevel_scores(id_imgs, np.arange(len(id_imgs)))

    out = {'seed': seed, 'T': T, 'n_id': int(len(id_imgs)), 'n_bank': int(len(bank_rows)), 'sources': {}}
    for src in np.unique(far_src):
        mk = np.flatnonzero(far_src == src)
        pick = rng.choice(mk, min(N_FAR_SUB, len(mk)), replace=False)
        far_imgs = np.asarray(Xfar[pick])
        f_far, L_far, _ = m.get_feats_logits(model, far_imgs, np.arange(len(far_imgs)),
                                             np.full(len(far_imgs), -1), device, use_amp)
        s_far = scores_from_feats(f_far, L_far, T, maha, bank, m.KNN_K)
        sat_far, grad_far = m.lowlevel_scores(far_imgs, np.arange(len(far_imgs)))

        matched = color_match(id_imgs, far_imgs)
        target_grad = float(np.median(grad_far))
        matched, t_kind, t_param, grad_after = transform_to_grad(matched, target_grad)
        f_m, L_m, _ = m.get_feats_logits(model, matched, np.arange(len(matched)), np.full(len(matched), -1),
                                         device, use_amp)
        s_m = scores_from_feats(f_m, L_m, T, maha, bank, m.KNN_K)
        sat_m, grad_m = m.lowlevel_scores(matched, np.arange(len(matched)))

        rec = {'n_far': int(len(far_imgs)), 'gray_set': is_gray_set(far_imgs),
               'transform': {'kind': t_kind, 'param': t_param},
               'grad_target': target_grad, 'grad_after': grad_after,
               'balance_sat_smd': smd(sat_m, sat_far), 'balance_grad_smd': smd(grad_m, grad_far),
               'balance_sat_smd_orig': smd(sat_id, sat_far), 'balance_grad_smd_orig': smd(grad_id, grad_far),
               'auc': {}}
        for k in ['energy_ts', 'msp_ts', 'maha', 'knn']:
            rec['auc'][k] = {
                'orig': auroc_id_vs_far(s_id[k], s_far[k]),
                'matched': auroc_id_vs_far(s_m[k], s_far[k]),
                'shift': float(roc_auc_score(np.r_[np.ones(len(s_id[k])), np.zeros(len(s_m[k]))],
                                             np.r_[s_id[k], s_m[k]])),
            }
        out['sources'][str(src)] = rec
        print(f'  seed {seed} {src}: energy orig {rec["auc"]["energy_ts"]["orig"]:.3f} | '
              f'khớp {rec["auc"]["energy_ts"]["matched"]:.3f} | dịch chuyển {rec["auc"]["energy_ts"]["shift"]:.3f} | '
              f'biến đổi {t_kind}({t_param}) | SMD độ dốc {rec["balance_grad_smd"]}', flush=True)
    out['seconds'] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'imagelevel_seed{seed}.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'seed {seed} xong trong {out["seconds"]} s', flush=True)
    return out


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if m.REQUIRE_GPU and device.type != 'cuda':
        raise SystemExit('KHÔNG THẤY GPU')
    print('device:', device, '| BACKBONE:', m.BACKBONE, '| N_ID_SUB:', N_ID_SUB, '| N_BANK:', N_BANK, flush=True)

    df = pd.read_csv(m.DATA_CSV)
    df = df[df['split'].astype(str).str.lower() == 'train'].reset_index(drop=True)
    names = df['bcn_filename'].tolist()
    rowmap = {f: i for i, f in enumerate(names)}
    X = np.memmap(IMG_MEMMAP, dtype=np.uint8, mode='r', shape=(len(names), m.RESIZE, m.RESIZE, 3))
    Xfar, far_src = m.build_far_array()
    if Xfar is None:
        raise SystemExit('Cần FAR_NPZ (hoặc FAR_ENABLE=1) để có bộ far-OOD.')
    seeds = [int(s) for s in m.SEEDS]
    for seed in seeds:
        run_seed(seed, X, rowmap, Xfar, far_src, device, use_amp)


if __name__ == '__main__':
    main()
