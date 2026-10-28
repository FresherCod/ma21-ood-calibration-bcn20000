# -*- coding: utf-8 -*-
"""
Mã 21: phát hiện OOD và hiệu chỉnh độ tin cậy trên BCN20000.

Thiết kế (xem STATUS.md, mục Quyết định mở):
- Class-holdout: lớp HOLDOUT (mặc định AK) không dùng để huấn luyện, dùng làm OOD.
- Chia ID theo lesion_id (StratifiedGroupKFold): train / val / id_test. Không có lesion chung.
- Val chỉ dùng cho early stopping và fit temperature. id_test và OOD chỉ dùng để đánh giá cuối.
- Đo calibration (ECE, NLL, Brier) trước và sau temperature scaling.
- Đo phát hiện OOD bằng MSP, energy và MC-dropout: AUROC và FPR@95TPR.
- Bootstrap theo lesion_id cho các chỉ số chính.

Cách chạy nhanh:
- Toàn bộ ảnh đã thu nhỏ về RESIZE px được đọc một lần vào RAM (uint8).
- Cắt ngẫu nhiên, lật và chuẩn hóa thực hiện trên GPU, nên CPU không phải xử lý từng ảnh.

Lưu trữ:
- Kaggle: checkpoint và kết quả nằm trong /kaggle/working/ma21_bcn20000 (thành Output của notebook).
  Muốn tiếp tục, gắn output lần trước làm Input; code tự khôi phục.
- Colab: checkpoint và kết quả trên Google Drive. Ảnh giải nén để ở ổ máy ảo.

Chạy thử cục bộ: đặt các biến môi trường (xem smoke test).
"""

# %% 1. Cấu hình và tải dữ liệu
import os
import sys
import json
import time
import hashlib
import zipfile
import subprocess
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd


def env(name, default, kind=str):
    v = os.environ.get(name)
    if v is None:
        return default
    if kind is bool:
        return v.strip().lower() in ('1', 'true', 'yes')
    if kind is list:
        return [x.strip() for x in v.split(',') if x.strip()]
    return kind(v)


IN_KAGGLE = os.path.isdir('/kaggle')
IN_COLAB = (not IN_KAGGLE) and os.path.isdir('/content')
INPUT_ROOT = env('INPUT_ROOT', '/kaggle/input')                        # output của lần chạy trước (gắn làm Input)
if IN_KAGGLE:
    DRIVE_ROOT = env('DRIVE_ROOT', '/kaggle/working/ma21_bcn20000')    # checkpoint và kết quả: thành Output của notebook
    LOCAL = env('LOCAL', '/tmp/ma21_local')                            # ảnh giải nén, zip: không lưu vào output
    DATA_DIR = LOCAL
elif IN_COLAB:
    DRIVE_ROOT = env('DRIVE_ROOT', '/content/drive/MyDrive/ma21_bcn20000')
    LOCAL = env('LOCAL', '/content/local')
    DATA_DIR = f'{DRIVE_ROOT}/data'
else:
    WORK = env('WORK', os.getcwd())
    DRIVE_ROOT = env('DRIVE_ROOT', f'{WORK}/ma21_bcn20000')
    LOCAL = env('LOCAL', f'{WORK}/local')
    DATA_DIR = f'{DRIVE_ROOT}/data'
ZIP_URL = 'https://ndownloader.figshare.com/files/45833178'   # BCN_20k_train.zip, 6368901290 byte
ZIP_MD5 = '2293c9c654a3f76d6506a57ad17cddd1'                  # checksum chính thức trên figshare
CSV_URL = 'https://ndownloader.figshare.com/files/49291684'   # bcn_20k_train.csv
HOLDOUT = env('HOLDOUT', ['AK'], list)
SEEDS = [int(s) for s in env('SEEDS', ['0'], list)]
EPOCHS = env('EPOCHS', 12, int)
BATCH = env('BATCH', 64, int)              # batch MỖI GPU; batch tổng = BATCH x số GPU (batch lớn để GPU không chờ CPU)
EVAL_BS = env('EVAL_BS', 128, int)         # batch khi suy luận (tổng)
LR = env('LR', 3e-4, float)
IMG = env('IMG', 224, int)
RESIZE = env('RESIZE', 256, int)
MC_PASSES = env('MC_PASSES', 20, int)
BOOT = env('BOOT', 500, int)
PATIENCE = env('PATIENCE', 3, int)
PRETRAINED = env('PRETRAINED', True, bool)
NUM_WORKERS = env('NUM_WORKERS', min(8, os.cpu_count() or 4), int)   # chỉ dùng khi thu nhỏ ảnh một lần
RESUME = env('RESUME', True, bool)                                     # tiếp tục từ checkpoint và bỏ qua seed đã xong
STOP_AFTER_EPOCH = env('STOP_AFTER_EPOCH', -1, int)                    # CHỈ để thử nghiệm: dừng sau epoch này
FORCE_GPU_DATA = env('FORCE_GPU_DATA', False, bool)                    # CHỈ để thử nghiệm: đường đi "ảnh trong tensor" trên CPU
RUN_TAG = env('RUN_TAG', 'v2')                 # đổi khi đổi cách chia dữ liệu, để không nạp nhầm checkpoint cũ
RUN_ROOT = f'{DRIVE_ROOT}/{RUN_TAG}'
FAR_NPZ = env('FAR_NPZ', [], list)                                     # bộ far-OOD dạng .npz (ví dụ MedMNIST); mỗi tệp một nguồn
FAR_KEY = env('FAR_KEY', 'test_images')                                # khóa mảng ảnh trong tệp .npz
GRAY_CONTROL = env('GRAY_CONTROL', True, bool)                         # đối chiếu: ảnh da đổi sang thang xám, cùng nội dung
REQUIRE_GPU = env('REQUIRE_GPU', IN_KAGGLE, bool)                      # trên Kaggle: dừng ngay nếu không thấy GPU, tránh chạy hàng giờ trên CPU
FAR_ENABLE = env('FAR_ENABLE', False, bool)                            # CHỈ bật sau khi người dùng duyệt tải
# Bộ far-OOD đề xuất: MedMNIST+ 224 px (Zenodo 10519652, CC BY 4.0), tập test. Không dùng DermaMNIST (CC BY-NC).
FAR_URLS = {
    'retinamnist_224.npz': 'https://zenodo.org/records/10519652/files/retinamnist_224.npz?download=1',      # 128,0 MB
    'pneumoniamnist_224.npz': 'https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1',  # 214,4 MB
    'breastmnist_224.npz': 'https://zenodo.org/records/10519652/files/breastmnist_224.npz?download=1',      # 30,9 MB
}
DATA_CSV = env('DATA_CSV', f'{DATA_DIR}/bcn_20k_train.csv')
IMAGE_DIR = env('IMAGE_DIR', f'{LOCAL}/train')
CKPT_DIR = env('CKPT_DIR', f'{RUN_ROOT}/checkpoints')
OUT = env('OUT', f'{RUN_ROOT}/outputs')


def md5_of(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def ensure_data():
    if 'DATA_CSV' in os.environ and 'IMAGE_DIR' in os.environ:
        print('Dùng dữ liệu cục bộ:', DATA_CSV, IMAGE_DIR)
        return
    if IN_COLAB and not os.path.isdir('/content/drive/MyDrive'):
        raise RuntimeError('Chưa gắn Google Drive. Chạy cell gắn Drive trước.')
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    if not Path(DATA_CSV).exists():
        subprocess.run(['wget', '-q', '-O', DATA_CSV, CSV_URL], check=True)

    marker = Path(LOCAL) / 'extracted.ok'
    if marker.exists() and Path(IMAGE_DIR).is_dir():
        print('Đã giải nén trên máy:', IMAGE_DIR)
        return

    zip_path = Path(DATA_DIR) / 'BCN_20k_train.zip'
    verified = Path(DATA_DIR) / 'zip_md5_ok.txt'
    already_ok = zip_path.exists() and verified.exists() and verified.read_text().strip() == ZIP_MD5
    if not already_ok:
        if not (zip_path.exists() and md5_of(zip_path) == ZIP_MD5):
            subprocess.run(['wget', '-c', '--progress=dot:giga', '-O', str(zip_path), ZIP_URL], check=True)
        got = md5_of(zip_path)
        if got != ZIP_MD5:
            raise RuntimeError(f'MD5 không khớp: {got} != {ZIP_MD5}. Xóa tệp zip và chạy lại.')
        verified.write_text(ZIP_MD5)
    print('MD5 khớp:', ZIP_MD5)

    Path(LOCAL).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(IMAGE_DIR)
    marker.write_text('ok')
    if IN_KAGGLE:
        zip_path.unlink()          # zip nằm ở /tmp, không cần giữ sau khi giải nén
    print('Giải nén xong:', IMAGE_DIR)


def restore_from_input():
    """Khôi phục checkpoint và kết quả của đúng RUN_TAG từ output lần trước (gắn làm Input trên Kaggle).
    Chỉ chạy khi thư mục checkpoint của RUN_TAG chưa có."""
    import glob
    import shutil
    if os.path.isdir(f'{RUN_ROOT}/checkpoints') or not os.path.isdir(INPUT_ROOT):
        return
    found = sorted(glob.glob(f'{INPUT_ROOT}/*/ma21_bcn20000/{RUN_TAG}/checkpoints') +
                   glob.glob(f'{INPUT_ROOT}/*/{RUN_TAG}/checkpoints'))
    if not found:
        # In các mục hiện có trong Input để biết đã Add Input chưa và đúng output chưa
        seen = sorted(str(p.relative_to(INPUT_ROOT)) for pat in ('*', '*/*', '*/*/*') for p in Path(INPUT_ROOT).glob(pat))
        print(f'Không có checkpoint {RUN_TAG} đầu vào; bắt đầu từ đầu.')
        print(f'Mục trong {INPUT_ROOT} (tối đa 40):', seen[:40] or 'trống, chưa Add Input')
        return
    src_root = Path(found[0]).parent
    for sub in ('checkpoints', 'outputs'):
        s = src_root / sub
        if s.is_dir():
            shutil.copytree(s, Path(RUN_ROOT) / sub, dirs_exist_ok=True)
    print('Đã khôi phục từ:', src_root)


# %% 2. Đọc ảnh vào RAM và chia dữ liệu theo lesion_id
def lesion_split(idd, seed):
    """Chia theo tổn thương, giữ tỉ lệ lớp: mỗi lớp, test ~20% và val ~10% số tổn thương (tối thiểu 1 nếu đủ lớn),
    phần còn lại là train. Một tổn thương chỉ nằm ở đúng một tập."""
    if (idd.groupby('lesion_id')['diagnosis'].nunique() > 1).any():
        raise ValueError('Có tổn thương mang nhiều chẩn đoán')
    les = idd.drop_duplicates('lesion_id')[['lesion_id', 'diagnosis']]
    rng = np.random.default_rng(seed)
    role = {}
    for _, grp in les.groupby('diagnosis'):
        ids = grp['lesion_id'].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_te = max(1, round(0.2 * n)) if n >= 2 else 0
        n_va = max(1, round(0.1 * n)) if n >= 4 else 0
        for i, lid in enumerate(ids):
            role[lid] = 'te' if i < n_te else ('va' if i < n_te + n_va else 'tr')
    r = idd['lesion_id'].map(role).to_numpy()
    return np.flatnonzero(r == 'tr'), np.flatnonzero(r == 'va'), np.flatnonzero(r == 'te')


def index_images(root):
    idx = {}
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx.setdefault(f, os.path.join(dp, f))
    return idx


def _decode_one(job):
    i, src = job
    im = Image.open(src)
    im.draft('RGB', (RESIZE * 2, RESIZE * 2))      # giải mã ở độ phân giải thấp, nhanh hơn
    im = im.convert('RGB').resize((RESIZE, RESIZE), Image.BILINEAR)
    return i, np.asarray(im, dtype=np.uint8)


def build_array():
    """Đọc mọi ảnh train một lần, thu nhỏ về RESIZE px, giữ trong RAM dạng uint8 (N, R, R, 3)."""
    from multiprocessing import Pool
    from PIL import Image as _Image  # noqa: F401  (đảm bảo PIL có sẵn cho worker)
    df = pd.read_csv(DATA_CSV)
    if 'split' in df.columns:
        df = df[df['split'].astype(str).str.lower() == 'train']
    names = df['bcn_filename'].tolist()
    idx = index_images(IMAGE_DIR)
    miss = [f for f in names if f not in idx]
    if miss:
        raise FileNotFoundError(f'Thiếu {len(miss)} ảnh, ví dụ: {miss[:3]}')
    X = np.empty((len(names), RESIZE, RESIZE, 3), dtype=np.uint8)
    jobs = [(i, idx[f]) for i, f in enumerate(names)]
    print(f'Đọc và thu nhỏ {len(jobs)} ảnh về {RESIZE} px vào RAM...')
    t0 = time.time()
    with Pool(max(1, NUM_WORKERS)) as p:
        for k, (i, arr) in enumerate(p.imap_unordered(_decode_one, jobs, chunksize=64)):
            X[i] = arr
            if (k + 1) % 2000 == 0:
                print('  đã xong', k + 1)
    print(f'Xong trong {time.time() - t0:.0f}s, RAM ~{X.nbytes / 1e9:.2f} GB')
    rowmap = {f: i for i, f in enumerate(names)}
    return X, rowmap


def ensure_far_files():
    """Tải các tệp far-OOD đề xuất (chỉ khi FAR_ENABLE=1 và chưa có FAR_NPZ). Trả về danh sách đường dẫn."""
    far_dir = Path(LOCAL) / 'far'
    far_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in FAR_URLS.items():
        target = far_dir / name
        if not target.exists():
            print(f'Tải far-OOD {name} ...')
            tmp = far_dir / (name + '.part')          # ghi vào tệp tạm rồi đổi tên, tránh tệp dở khi nhiều tiến trình chạy cùng lúc
            subprocess.run(['wget', '-q', '-O', str(tmp), url], check=True)
            os.replace(tmp, target)
        paths.append(str(target))
    return paths


def build_far_array():
    """Đọc các bộ far-OOD (.npz), đưa về RGB và RESIZE px. Trả về (N, R, R, 3) uint8 và tên nguồn của từng ảnh."""
    paths = list(FAR_NPZ)
    if not paths and FAR_ENABLE:
        paths = ensure_far_files()
    if not paths:
        return None, None
    chunks, names = [], []
    for path in paths:
        src = Path(path).stem
        arr = np.load(path)[FAR_KEY]
        out = np.empty((len(arr), RESIZE, RESIZE, 3), dtype=np.uint8)
        for i, a in enumerate(arr):
            if a.ndim == 2:                                  # ảnh xám: lặp thành 3 kênh
                a = np.stack([a, a, a], axis=-1)
            im = Image.fromarray(a.astype(np.uint8)).convert('RGB').resize((RESIZE, RESIZE), Image.BILINEAR)
            out[i] = np.asarray(im, dtype=np.uint8)
        chunks.append(out)
        names += [src] * len(out)
        print(f'far-OOD {src}: {len(out)} ảnh (từ {path})')
    return np.concatenate(chunks), np.array(names)


def build_split(seed, rowmap):
    df = pd.read_csv(DATA_CSV)
    if 'split' in df.columns:
        df = df[df['split'].astype(str).str.lower() == 'train']
    df = df.reset_index(drop=True)
    df['row'] = df['bcn_filename'].map(rowmap).astype(int)

    ood = df[df['diagnosis'].isin(HOLDOUT)].reset_index(drop=True)
    idd = df[~df['diagnosis'].isin(HOLDOUT)].reset_index(drop=True)
    classes = sorted(idd['diagnosis'].unique().tolist())
    cmap = {c: i for i, c in enumerate(classes)}
    y = idd['diagnosis'].map(cmap).to_numpy()
    g = idd['lesion_id'].to_numpy()

    tr, va, te = lesion_split(idd, seed)

    gs_tr, gs_va, gs_te = set(g[tr]), set(g[va]), set(g[te])
    assert not (gs_tr & gs_va) and not (gs_tr & gs_te) and not (gs_va & gs_te), 'Rò rỉ lesion giữa các tập'

    return dict(idd=idd, ood=ood, y=y, g=g, classes=classes, tr=tr, va=va, te=te)


# %% 3. Mô hình và huấn luyện (có checkpoint và tiếp tục)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_rng():
    st = {'torch': torch.get_rng_state(), 'np': np.random.get_state(), 'py': random.getstate()}
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st):
    torch.set_rng_state(st['torch'])
    np.random.set_state(st['np'])
    random.setstate(st['py'])
    if 'cuda' in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])


def save_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)          # ghi xong mới thay tệp cũ, tránh tệp dở khi mất kết nối


def build_model(n_cls):
    if PRETRAINED:
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, n_cls))
    return m


def gpu_prep(xb, device, train):
    """xb: uint8 (n, R, R, 3) trên CPU. Cắt, lật, chuẩn hóa trên GPU."""
    x = xb.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    n = x.shape[0]
    if train:
        # Cắt ngẫu nhiên từng ảnh bằng chỉ số, không dùng vòng lặp Python (một lệnh cho cả batch).
        oy = torch.randint(0, RESIZE - IMG + 1, (n,), device=device)
        ox = torch.randint(0, RESIZE - IMG + 1, (n,), device=device)
        ar = torch.arange(IMG, device=device)
        rows = (oy[:, None] + ar)[:, None, :, None]           # (n, 1, IMG, 1)
        cols = (ox[:, None] + ar)[:, None, None, :]           # (n, 1, 1, IMG)
        bi = torch.arange(n, device=device)[:, None, None, None]
        ci = torch.arange(3, device=device)[None, :, None, None]
        out = x[bi, ci, rows, cols]                           # (n, 3, IMG, IMG)
        hf = (torch.rand(n, device=device) < 0.5).view(-1, 1, 1, 1)
        vf = (torch.rand(n, device=device) < 0.5).view(-1, 1, 1, 1)
        out = torch.where(hf, out.flip(3), out)
        out = torch.where(vf, out.flip(2), out)
    else:
        o = (RESIZE - IMG) // 2
        out = x[:, :, o:o + IMG, o:o + IMG]
    return (out - MEAN.to(device)) / STD.to(device)


def batches(X, rows, labels, bs, device, order=None):
    """X: tensor uint8 (N, R, R, 3) đã nằm trên GPU, hoặc numpy trên CPU."""
    idx = np.arange(len(rows)) if order is None else order
    if torch.is_tensor(X):
        # Mọi chỉ số đặt sẵn trên GPU: không còn copy CPU->GPU ở mỗi bước.
        dev = X.device
        order_t = torch.as_tensor(idx, device=dev)
        rows_t = torch.as_tensor(rows, device=dev)
        labels_t = torch.as_tensor(labels, device=dev)
        for s in range(0, len(idx), bs):
            b = order_t[s:s + bs]
            yield X[rows_t[b]], labels_t[b]
        return
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        xb = torch.from_numpy(X[rows[b]])
        if device.type == 'cuda':
            xb = xb.pin_memory()
        yield xb, torch.as_tensor(labels[b])


@torch.no_grad()
def get_feats_logits(model, X, rows, labels, device, use_amp, bs=None):
    model.eval()
    feats, ys = [], []
    for xb, yb in batches(X, rows, labels, bs or EVAL_BS, device):
        x = gpu_prep(xb, device, train=False)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            f = torch.flatten(model.avgpool(model.features(x)), 1)
        feats.append(f.float().cpu())
        ys.append(yb.cpu())
    feats = torch.cat(feats)
    ys = torch.cat(ys)
    logits = model.classifier(feats.to(device)).float().cpu()
    return feats, logits, ys


def train_model(seed, X, tr_rows, tr_y, va_rows, va_y, n_cls, class_w, device, use_amp):
    set_seed(seed)
    model = build_model(n_cls).to(device)
    ck_dir = Path(CKPT_DIR) / f'seed{seed}'
    best_path = ck_dir / 'best_model.pt'
    last_path = ck_dir / 'last.pt'
    if RESUME and best_path.exists():
        ck = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state'])
        print(f'  đã có {best_path.name}, bỏ qua huấn luyện')
        return model, ck['hist'], ck['best_ep']

    # Dùng tất cả GPU có sẵn: DataParallel chia batch ra từng card (chỉ dùng cho forward khi huấn luyện).
    ngpu = torch.cuda.device_count() if device.type == 'cuda' else 1
    net = nn.DataParallel(model) if ngpu > 1 else model
    gbs = BATCH * ngpu
    steps_per_epoch = -(-len(tr_rows) // gbs)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    crit = nn.CrossEntropyLoss(weight=class_w.to(device))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best = (float('inf'), None, -1)
    bad, hist, start = 0, [], 0
    if RESUME and last_path.exists():
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        if use_amp and ck['scaler'] is not None:
            scaler.load_state_dict(ck['scaler'])
        best = (ck['best_loss'], ck['best_state'], ck['best_ep'])
        bad, hist, start = ck['bad'], ck['hist'], ck['epoch'] + 1
        restore_rng(ck['rng'])
        if bad >= PATIENCE:
            start = EPOCHS            # đã dừng sớm trước khi ngắt
        print(f'  tiếp tục từ epoch {start} (đọc {last_path.name})')

    for ep in range(start, EPOCHS):
        t0 = time.time()
        model.train()
        # Thứ tự mẫu chỉ phụ thuộc (seed, epoch) nên khi tiếp tục vẫn giống lần chạy không ngắt.
        order = np.random.default_rng(seed * 100003 + ep).permutation(len(tr_rows))
        tot, n = torch.zeros((), device=device), 0
        for xb, yb in batches(X, tr_rows, tr_y, gbs, device, order):
            x = gpu_prep(xb, device, train=True)
            y = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = crit(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.detach() * len(y)          # cộng dồn trên GPU, không đồng bộ mỗi bước
            n += len(y)
        _, vlog, vy = get_feats_logits(model, X, va_rows, va_y, device, use_amp)
        train_loss = float(tot.item()) / n
        vnll = F.cross_entropy(vlog, vy).item()
        hist.append(dict(epoch=ep, train_loss=train_loss, val_nll=vnll, sec=round(time.time() - t0, 1),
                         lr=float(sched.get_last_lr()[0]),
                         gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == 'cuda' else None))
        mem = f' | GPU đỉnh {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB' if device.type == 'cuda' else ''
        print(f'  epoch {ep:02d} train_loss {train_loss:.4f} val_nll {vnll:.4f} ({hist[-1]["sec"]}s){mem}')
        if vnll < best[0] - 1e-4:
            best = (vnll, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1

        save_atomic({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                     'sched': sched.state_dict(), 'scaler': scaler.state_dict() if use_amp else None,
                     'best_loss': best[0], 'best_state': best[1], 'best_ep': best[2],
                     'bad': bad, 'hist': hist, 'rng': get_rng(), 'seed': seed, 'epochs': EPOCHS}, last_path)
        if STOP_AFTER_EPOCH == ep:
            raise SystemExit(f'[thử nghiệm] dừng sau epoch {ep}, checkpoint đã lưu')
        if bad >= PATIENCE:
            print('  dừng sớm')
            break

    model.load_state_dict(best[1])
    save_atomic({'state': best[1], 'hist': hist, 'best_ep': best[2]}, best_path)
    return model, hist, best[2]


def fit_temperature(logits, ys):
    logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logt.exp(), ys)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logt.exp().item())


@torch.no_grad()
def mc_probs(model, feats, T_, passes, device):
    head = model.classifier
    head.eval()
    head[0].train()                      # chỉ bật dropout, giữ BatchNorm ở chế độ eval
    f = feats.to(device)
    ps = [torch.softmax(head(f) / T_, dim=1).cpu() for _ in range(passes)]
    head.eval()
    return torch.stack(ps).mean(0)


# %% 4. Chỉ số
def ece_score(P, ys, n_bins=15):
    conf, pred = P.max(1)
    acc = (pred == ys).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return float(e)


def nll(P, ys):
    return float(-torch.log(P[torch.arange(len(ys)), ys].clamp_min(1e-12)).mean().item())


def brier(P, ys):
    oh = F.one_hot(ys, P.shape[1]).float()
    return float(((P - oh) ** 2).sum(1).mean().item())


def clf_metrics(P, ys):
    pred = P.argmax(1).numpy()
    yy = ys.numpy()
    return dict(acc=float((pred == yy).mean()), bal_acc=float(balanced_accuracy_score(yy, pred)),
                nll=nll(P, ys), brier=brier(P, ys), ece=ece_score(P, ys))


def auroc_fpr(sid, sood):
    yy = np.r_[np.ones(len(sid)), np.zeros(len(sood))]
    s = np.r_[sid, sood]
    auc = float(roc_auc_score(yy, s))
    thr = np.percentile(sid, 5)                     # giữ 95% ID
    fpr = float((sood >= thr).mean())
    return auc, fpr


def lesion_bootstrap(metric_fn, gid, good, B, seed):
    rng = np.random.default_rng(seed)
    ug_i, ug_o = np.unique(gid), np.unique(good)
    mi = {k: np.flatnonzero(gid == k) for k in ug_i}
    mo = {k: np.flatnonzero(good == k) for k in ug_o}
    vals = []
    for _ in range(B):
        ii = np.concatenate([mi[k] for k in rng.choice(ug_i, len(ug_i))])
        oo = np.concatenate([mo[k] for k in rng.choice(ug_o, len(ug_o))])
        vals.append(metric_fn(ii, oo))
    out = {}
    for k in vals[0]:
        arr = np.array([v[k] for v in vals])
        out[k] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    return out, pd.DataFrame(vals)


def env_info():
    info = dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
                cuda=torch.cuda.is_available(), cpu_count=os.cpu_count(), in_kaggle=IN_KAGGLE, in_colab=IN_COLAB)
    if torch.cuda.is_available():
        info['gpu'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info['n_gpu'] = torch.cuda.device_count()
    return info


CONFIG_KEYS = ['SEEDS', 'HOLDOUT', 'EPOCHS', 'BATCH', 'EVAL_BS', 'LR', 'IMG', 'RESIZE', 'MC_PASSES', 'BOOT',
               'PATIENCE', 'PRETRAINED', 'NUM_WORKERS', 'RESUME', 'DATA_CSV', 'IMAGE_DIR', 'CKPT_DIR', 'OUT',
               'ZIP_URL', 'CSV_URL', 'RUN_TAG', 'FAR_NPZ', 'FAR_KEY', 'GRAY_CONTROL']


def reliability(P, ys, n_bins=15):
    """Dữ liệu biểu đồ độ tin cậy: theo từng bin, số mẫu, độ chính xác và độ tin cậy trung bình."""
    conf, pred = P.max(1)
    c, a = conf.numpy(), (pred == ys).float().numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi)
        rows.append(dict(bin_lo=round(float(lo), 4), bin_hi=round(float(hi), 4), count=int(m.sum()),
                         accuracy=float(a[m].mean()) if m.any() else None,
                         mean_confidence=float(c[m].mean()) if m.any() else None))
    return pd.DataFrame(rows)


def write_details(seed, S, hist, best_ep, T_, Lva, yva, Lte, yte, Lood, pr_te, pt_te,
                  boot_cal, boot_ood, res, t_seed0):
    """Lưu toàn bộ dữ liệu trung gian của một seed để kiểm tra và phân tích lại mà không cần chạy lại."""
    import shutil
    import sklearn
    import PIL
    from sklearn.metrics import confusion_matrix, classification_report
    d = Path(OUT) / f'seed{seed}'
    d.mkdir(parents=True, exist_ok=True)
    idd, ood, tr, va, te = S['idd'], S['ood'], S['tr'], S['va'], S['te']
    classes = S['classes']

    # 1. Ảnh nào thuộc tập nào (một dòng mỗi ảnh)
    split_arr = np.empty(len(idd), dtype=object)
    split_arr[tr] = 'train'
    split_arr[va] = 'val'
    split_arr[te] = 'id_test'
    sp = pd.concat([
        pd.DataFrame({'bcn_filename': idd['bcn_filename'], 'lesion_id': idd['lesion_id'],
                      'diagnosis': idd['diagnosis'], 'row': idd['row'], 'split': split_arr,
                      'y': S['y'], 'is_ood': 0}),
        pd.DataFrame({'bcn_filename': ood['bcn_filename'], 'lesion_id': ood['lesion_id'],
                      'diagnosis': ood['diagnosis'], 'row': ood['row'], 'split': 'ood',
                      'y': -1, 'is_ood': 1}),
    ], ignore_index=True)
    sp.to_csv(d / 'splits.csv', index=False)

    # 2. Số ảnh và số tổn thương theo tập và theo lớp
    cc = []
    for name, idx in [('train', tr), ('val', va), ('id_test', te)]:
        for ci, c in enumerate(classes):
            m = S['y'][idx] == ci
            cc.append(dict(split=name, diagnosis=c, images=int(m.sum()),
                           lesions=len(set(idd['lesion_id'].values[idx][m]))))
    cc.append(dict(split='ood', diagnosis=','.join(HOLDOUT), images=len(ood), lesions=ood['lesion_id'].nunique()))
    pd.DataFrame(cc).to_csv(d / 'class_counts.csv', index=False)

    # 3. Lịch sử huấn luyện từng epoch
    pd.DataFrame(hist).to_csv(d / 'history.csv', index=False)

    # 4. Temperature scaling
    t_info = dict(temperature=T_, best_epoch=best_ep, n_val=int(len(yva)),
                  val_nll_raw=nll(torch.softmax(Lva, 1), yva), val_nll_ts=nll(torch.softmax(Lva / T_, 1), yva))
    with open(d / 'temperature.json', 'w', encoding='utf-8') as f:
        json.dump(t_info, f, ensure_ascii=False, indent=2)

    # 5. Độ tin cậy theo bin, ma trận nhầm lẫn, chỉ số theo lớp (trước và sau TS)
    reliability(pr_te, yte).to_csv(d / 'reliability_id_test_raw.csv', index=False)
    reliability(pt_te, yte).to_csv(d / 'reliability_id_test_ts.csv', index=False)
    labels = list(range(len(classes)))
    for tag, P in [('raw', pr_te), ('ts', pt_te)]:
        pred, yy = P.argmax(1).numpy(), yte.numpy()
        cm = confusion_matrix(yy, pred, labels=labels)
        pd.DataFrame(cm, index=[f'true_{c}' for c in classes],
                     columns=[f'pred_{c}' for c in classes]).to_csv(d / f'confusion_id_test_{tag}.csv')
        rep = classification_report(yy, pred, labels=labels, target_names=classes,
                                    output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(d / f'per_class_id_test_{tag}.csv')

    # 6. Logits thô để phân tích lại
    np.savez_compressed(d / 'logits.npz', Lva=Lva.numpy(), yva=yva.numpy(), Lte=Lte.numpy(),
                        yte=yte.numpy(), Lood=Lood.numpy(), T=np.array(T_))

    # 7. Toàn bộ mẫu bootstrap (không chỉ khoảng tin cậy)
    boot_cal.to_csv(d / 'bootstrap_calibration_ts.csv', index_label='replicate')
    boot_ood.to_csv(d / 'bootstrap_ood.csv', index_label='replicate')

    # 8. Thông tin tái lập: cấu hình, phiên bản, mã băm của script, GPU
    src = globals().get('__file__')
    script_sha = None
    if src and os.path.exists(src):
        script_sha = hashlib.sha256(open(src, 'rb').read()).hexdigest()
        shutil.copy(src, d / 'code_snapshot_ood_calibration.py')
    info = dict(
        seed=seed, finished_at=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_sec=round(time.time() - t_seed0, 1),
        script_sha256=script_sha, zip_md5_expected=ZIP_MD5,
        config={k: globals().get(k) for k in CONFIG_KEYS},
        versions=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                      numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                      pillow=PIL.__version__),
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None,
        n_images={'train': int(len(tr)), 'val': int(len(va)), 'id_test': int(len(te)), 'ood': int(len(ood))},
        classes=classes, class_index={c: i for i, c in enumerate(classes)}, holdout=HOLDOUT,
        summary={k: res[k] for k in ['temperature', 'best_epoch', 'id_metrics_raw', 'id_metrics_ts', 'ood']},
    )
    with open(d / 'run_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    print(f'  đã lưu chi tiết vào {d}')


# %% 5. Chạy một seed
def to_gray_batch(X, rows):
    """Lấy ảnh theo hàng, đổi sang thang xám (3 kênh giống nhau), giữ nguyên kích thước và dtype uint8."""
    if torch.is_tensor(X):
        xb = X[torch.as_tensor(rows, device=X.device)]
    else:
        xb = torch.from_numpy(X[rows])
    g = (0.299 * xb[..., 0].float() + 0.587 * xb[..., 1].float() + 0.114 * xb[..., 2].float())
    g = g.round().clamp(0, 255).to(torch.uint8)
    return g.unsqueeze(-1).repeat(1, 1, 1, 3)


def image_bootstrap(sid, sfar, B, seed):
    """Khoảng tin cậy 95% của AUROC và FPR95, bootstrap theo ảnh (ảnh far-OOD không gom theo tổn thương)."""
    rng = np.random.default_rng(seed)
    au, fp = [], []
    for _ in range(B):
        i = rng.integers(0, len(sid), len(sid))
        j = rng.integers(0, len(sfar), len(sfar))
        a, f = auroc_fpr(sid[i], sfar[j])
        au.append(a)
        fp.append(f)
    return ([float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))],
            [float(np.percentile(fp, 2.5)), float(np.percentile(fp, 97.5))])


def run_seed(seed, device, use_amp, X, rowmap, Xfar=None, far_src=None):
    t_seed0 = time.time()
    res_path = Path(OUT) / f'results_seed{seed}.json'
    if RESUME and res_path.exists():
        prev = json.load(open(res_path, encoding='utf-8'))
        if not GRAY_CONTROL or 'gray_control' in prev:
            print(f'seed {seed}: đã có {res_path.name}, bỏ qua')
            return prev
        print(f'seed {seed}: thiếu đối chứng xám, tính lại phần đánh giá từ checkpoint (không huấn luyện lại)')

    S = build_split(seed, rowmap)
    idd, ood, y = S['idd'], S['ood'], S['y']
    K = len(S['classes'])
    tr, va, te = S['tr'], S['va'], S['te']
    print(f'seed {seed}: train {len(tr)}, val {len(va)}, id_test {len(te)}, ood {len(ood)}, lớp ID {S["classes"]}')

    counts = np.bincount(y[tr], minlength=K)
    class_w = torch.tensor(len(tr) / (K * np.maximum(counts, 1)), dtype=torch.float32)

    row_id = idd['row'].values
    row_ood = ood['row'].values
    model, hist, best_ep = train_model(seed, X, row_id[tr], y[tr], row_id[va], y[va],
                                       K, class_w, device, use_amp)
    print(f'  best epoch {best_ep}')

    fva, Lva, yva = get_feats_logits(model, X, row_id[va], y[va], device, use_amp)
    T_ = fit_temperature(Lva, yva)
    print(f'  temperature T = {T_:.4f}')

    fte, Lte, yte = get_feats_logits(model, X, row_id[te], y[te], device, use_amp)
    food, Lood, _ = get_feats_logits(model, X, row_ood, np.full(len(ood), -1), device, use_amp)

    pr_te = torch.softmax(Lte, 1)
    pt_te = torch.softmax(Lte / T_, 1)
    torch.manual_seed(seed)                       # MC-dropout lặp lại được giữa các lần tính lại
    mc_te = mc_probs(model, fte, T_, MC_PASSES, device)
    mc_ood = mc_probs(model, food, T_, MC_PASSES, device)

    fns = {
        'msp_raw': lambda L: torch.softmax(L, 1).max(1).values,
        'msp_ts': lambda L: torch.softmax(L / T_, 1).max(1).values,
        'energy_raw': lambda L: torch.logsumexp(L, 1),
        'energy_ts': lambda L: T_ * torch.logsumexp(L / T_, 1),
    }
    sc_id = {k: f(Lte).numpy() for k, f in fns.items()}
    sc_ood = {k: f(Lood).numpy() for k, f in fns.items()}
    sc_id['mc_msp'] = mc_te.max(1).values.numpy()
    sc_ood['mc_msp'] = mc_ood.max(1).values.numpy()

    ood_res = {k: dict(zip(['auroc', 'fpr95'], auroc_fpr(sc_id[k], sc_ood[k]))) for k in sc_id}

    # Far-OOD: các bộ ảnh khác hẳn phương thức/nội dung, đánh giá cùng mô hình, cùng T và cùng điểm số
    far_res, sc_far = None, None
    if Xfar is not None:
        ffar, Lfar, _ = get_feats_logits(model, Xfar, np.arange(len(far_src)), np.full(len(far_src), -1),
                                         device, use_amp)
        torch.manual_seed(seed + 1)
        mc_far = mc_probs(model, ffar, T_, MC_PASSES, device)
        sc_far = {k: f(Lfar).numpy() for k, f in fns.items()}
        sc_far['mc_msp'] = mc_far.max(1).values.numpy()
        far_res = {}
        for src in np.unique(far_src):
            m = far_src == src
            far_res[str(src)] = {'n': int(m.sum())}
            for k in sc_far:
                a, f = auroc_fpr(sc_id[k], sc_far[k][m])
                ci_a, ci_f = image_bootstrap(sc_id[k], sc_far[k][m], BOOT, seed + 2)
                far_res[str(src)][k] = dict(auroc=a, fpr95=f, auroc_ci=ci_a, fpr95_ci=ci_f)
        print('  far-OOD (AUROC, FPR95, energy_ts):',
              {s_: (round(v['energy_ts']['auroc'], 4), round(v['energy_ts']['fpr95'], 4)) for s_, v in far_res.items()})

    # Đối chiếu xám: cùng ảnh da trong tập ID test, đổi sang thang xám (không đổi nội dung).
    # Nếu điểm OOD thay đổi mạnh chỉ vì màu sắc, đây là nguồn nhiễu trong kết quả far-OOD.
    gray_res, sc_gray = None, None
    if GRAY_CONTROL:
        Gx = to_gray_batch(X, row_id[te])
        _, Lg, _ = get_feats_logits(model, Gx, np.arange(len(te)), np.full(len(te), -1), device, use_amp)
        sc_gray = {k: f(Lg).numpy() for k, f in fns.items()}
        gray_res = {'n': int(len(te))}
        for k in sc_gray:
            a, f = auroc_fpr(sc_id[k], sc_gray[k])
            ci_a, ci_f = image_bootstrap(sc_id[k], sc_gray[k], BOOT, seed + 3)
            gray_res[k] = dict(auroc=a, fpr95=f, auroc_ci=ci_a, fpr95_ci=ci_f,
                               median_id=float(np.median(sc_id[k])), median_gray=float(np.median(sc_gray[k])))
        print('  đối chứng xám (AUROC, FPR95, energy_ts):',
              round(gray_res['energy_ts']['auroc'], 4), round(gray_res['energy_ts']['fpr95'], 4))

    gid = S['g'][te]
    good = ood['lesion_id'].values

    ci_cal, boot_cal = lesion_bootstrap(
        lambda ii, oo: dict(ece_ts=ece_score(pt_te[torch.as_tensor(ii)], yte[torch.as_tensor(ii)]),
                            nll_ts=nll(pt_te[torch.as_tensor(ii)], yte[torch.as_tensor(ii)])),
        gid, good, BOOT, seed)
    ci_ood, boot_ood = lesion_bootstrap(
        lambda ii, oo: {f'{m}_{k}': v for k in sc_id for m, v in zip(
            ['auroc', 'fpr95'], auroc_fpr(sc_id[k][ii], sc_ood[k][oo]))},
        gid, good, BOOT, seed + 1)

    res = dict(
        seed=seed, classes=S['classes'], holdout=HOLDOUT, n_train=int(len(tr)), n_val=int(len(va)),
        n_id_test=int(len(te)), n_ood=int(len(ood)), n_lesions_train=int(len(set(S['g'][tr]))),
        best_epoch=best_ep, temperature=T_, history=hist,
        id_metrics_raw=clf_metrics(pr_te, yte), id_metrics_ts=clf_metrics(pt_te, yte),
        ood=ood_res, ci_calibration_ts=ci_cal, ci_ood=ci_ood,
        env=env_info(), config=dict(EPOCHS=EPOCHS, BATCH_PER_GPU=BATCH, EVAL_BS=EVAL_BS, LR=LR, IMG=IMG,
                                    RESIZE=RESIZE, MC_PASSES=MC_PASSES, BOOT=BOOT, PATIENCE=PATIENCE,
                                    PRETRAINED=PRETRAINED, HOLDOUT=HOLDOUT),
    )
    os.makedirs(OUT, exist_ok=True)
    if far_res is not None:
        res['far_ood'] = far_res
    if gray_res is not None:
        res['gray_control'] = gray_res
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    pred_te = pt_te.argmax(1).numpy()
    df_id = pd.DataFrame({
        'bcn_filename': idd['bcn_filename'].values[te], 'lesion_id': gid,
        'diagnosis': idd['diagnosis'].values[te], 'is_ood': 0, 'y': yte.numpy(),
        'pred_raw': pr_te.argmax(1).numpy(), 'pred_ts': pred_te,
        'pmax_raw': pr_te.max(1).values.numpy(), 'pmax_ts': pt_te.max(1).values.numpy(),
        'mc_msp': sc_id['mc_msp'], 'energy_ts': sc_id['energy_ts']})
    df_ood = pd.DataFrame({
        'bcn_filename': ood['bcn_filename'].values, 'lesion_id': good,
        'diagnosis': ood['diagnosis'].values, 'is_ood': 1, 'y': -1,
        'pred_raw': torch.softmax(Lood, 1).argmax(1).numpy(),
        'pred_ts': torch.softmax(Lood / T_, 1).argmax(1).numpy(),
        'pmax_raw': torch.softmax(Lood, 1).max(1).values.numpy(),
        'pmax_ts': torch.softmax(Lood / T_, 1).max(1).values.numpy(),
        'mc_msp': sc_ood['mc_msp'], 'energy_ts': sc_ood['energy_ts']})
    # Xác suất từng lớp (trước và sau TS) cho cả ảnh ID và OOD
    pr_ood, pt_ood = torch.softmax(Lood, 1), torch.softmax(Lood / T_, 1)
    for k, c in enumerate(S['classes']):
        df_id[f'p_raw_{c}'] = pr_te[:, k].numpy()
        df_id[f'p_ts_{c}'] = pt_te[:, k].numpy()
        df_ood[f'p_raw_{c}'] = pr_ood[:, k].numpy()
        df_ood[f'p_ts_{c}'] = pt_ood[:, k].numpy()
    for k in ['msp_raw', 'msp_ts', 'energy_raw', 'energy_ts', 'mc_msp']:
        df_id[k] = sc_id[k]
        df_ood[k] = sc_ood[k]
    pd.concat([df_id, df_ood]).to_csv(os.path.join(OUT, f'predictions_seed{seed}.csv'), index=False)

    if far_res is not None:
        pd.DataFrame({'source': far_src, **{k: sc_far[k] for k in sc_far}}).to_csv(
            os.path.join(OUT, f'far_scores_seed{seed}.csv'), index=False)
    if gray_res is not None:
        pd.DataFrame({'bcn_filename': idd['bcn_filename'].values[te],
                      **{f'orig_{k}': sc_id[k] for k in fns},
                      **{f'gray_{k}': sc_gray[k] for k in fns}}).to_csv(
            os.path.join(OUT, f'gray_control_seed{seed}.csv'), index=False)
    write_details(seed, S, hist, best_ep, T_, Lva, yva, Lte, yte, Lood, pr_te, pt_te,
                  boot_cal, boot_ood, res, t_seed0)
    print('  ID test (sau TS):', {k: round(v, 4) for k, v in res['id_metrics_ts'].items()})
    print('  OOD (AUROC, FPR95):', {k: (round(v['auroc'], 4), round(v['fpr95'], 4)) for k, v in ood_res.items()})
    return res


# %% 6. Tổng hợp và chạy
def summarize():
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, 'results_seed*.json'))):
        r = json.load(open(p, encoding='utf-8'))
        row = {'seed': r['seed'], 'T': r['temperature'], 'best_epoch': r['best_epoch']}
        for k, v in r['id_metrics_raw'].items():
            row[f'raw_{k}'] = v
        for k, v in r['id_metrics_ts'].items():
            row[f'ts_{k}'] = v
        for k, v in r['ood'].items():
            row[f'auroc_{k}'] = v['auroc']
            row[f'fpr95_{k}'] = v['fpr95']
        for src, d in r.get('far_ood', {}).items():
            for k, v in d.items():
                if isinstance(v, dict):
                    row[f'far_{src}_auroc_{k}'] = v['auroc']
                    row[f'far_{src}_fpr95_{k}'] = v['fpr95']
        for k, v in r.get('gray_control', {}).items():
            if isinstance(v, dict):
                row[f'gray_{k}_auroc'] = v['auroc']
                row[f'gray_{k}_fpr95'] = v['fpr95']
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, 'summary_per_seed.csv'), index=False)
        print(df.T.to_string())
        print('\nTrung bình ± độ lệch chuẩn qua các seed:')
        print(df.drop(columns=['seed']).agg(['mean', 'std']).T.to_string())


def main():
    os.makedirs(OUT, exist_ok=True)
    if env('SUMMARIZE_ONLY', False, bool):
        summarize()
        return
    restore_from_input()                       # phải chạy trước khi tạo CKPT_DIR
    os.makedirs(CKPT_DIR, exist_ok=True)
    ensure_data()
    if env('SETUP_ONLY', False, bool):
        if FAR_ENABLE and not FAR_NPZ:
            ensure_far_files()                 # tải far-OOD một lần ở đây, các tiến trình con chỉ đọc
        print('Đã chuẩn bị dữ liệu.')
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if REQUIRE_GPU and device.type != 'cuda':
        raise SystemExit('KHÔNG THẤY GPU (device=cpu). Dừng để tránh chạy chậm hàng giờ. '
                         'Kiểm tra Settings → Accelerator và số GPU của phiên.')
    print('device:', device, '| torch:', torch.__version__, '| holdout:', HOLDOUT, '| seeds:', SEEDS)
    print('checkpoint:', CKPT_DIR, '| kết quả:', OUT)
    X, rowmap = build_array()
    Xfar, far_src = build_far_array()
    if device.type == 'cuda' or FORCE_GPU_DATA:
        # Cả bộ ảnh uint8 (~2,4 GB) đưa lên GPU một lần, để mỗi batch được lấy trực tiếp trên GPU.
        X = torch.from_numpy(X).to(device)
        print(f'Đã đưa ảnh lên {device}: {X.numel() / 1e9:.2f} GB uint8')
        if Xfar is not None:
            Xfar = torch.from_numpy(Xfar).to(device)
    for s in SEEDS:
        run_seed(s, device, use_amp, X, rowmap, Xfar, far_src)
    if env('SUMMARIZE', True, bool):          # tiến trình con của bộ chạy song song bỏ qua bước này
        summarize()


if __name__ == '__main__':
    main()
