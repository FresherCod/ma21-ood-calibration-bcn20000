# -*- coding: utf-8 -*-
"""
Chạy song song nhiều seed, mỗi GPU một tiến trình riêng (không dùng DataParallel).

Trình tự:
1. Chuẩn bị dữ liệu một lần (chọn CSV và zip trong Input, kiểm MD5, ghi manifest) để các tiến trình không tranh nhau.
2. Chia seed theo vòng tròn cho từng GPU, mỗi GPU một tiến trình chạy tuần tự các seed của nó.
3. Theo dõi tiến độ mỗi phút, rồi tổng hợp kết quả.

Mặc định dùng SEEDS trong môi trường; SCRIPT_PATH trỏ tới ood_calibration.py.
Biến N_GPU chỉ để thử nghiệm trên CPU (giả lập số GPU).
"""
import os
import sys
import time
import subprocess
from pathlib import Path

import torch

SCRIPT = os.environ.get('SCRIPT_PATH', '/tmp/ma21_code/ood_calibration.py')
LOG_DIR = os.environ.get('LOG_DIR', '/kaggle/working/ma21_bcn20000/logs')
SEEDS = [s.strip() for s in os.environ.get('SEEDS', '0').split(',') if s.strip()]
N_GPU = int(os.environ.get('N_GPU', torch.cuda.device_count()))
REQUIRE_GPU = os.environ.get('REQUIRE_GPU', '1' if os.path.isdir('/kaggle') else '0').strip().lower() in ('1', 'true', 'yes')
if REQUIRE_GPU and N_GPU < 1:
    # Kiểm tra trước bước chuẩn bị dữ liệu: nếu không có GPU thì dừng ngay, không tải và giải nén ~6 GB vô ích.
    raise SystemExit('KHÔNG THẤY GPU (N_GPU=0). Dừng trước khi chuẩn bị dữ liệu. '
                     'Kiểm tra Settings → Accelerator và số GPU của phiên.')

n_proc = min(max(1, N_GPU), len(SEEDS))
groups = [SEEDS[i::n_proc] for i in range(n_proc)]
common = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
if 'NUM_WORKERS' not in os.environ:
    common['NUM_WORKERS'] = str(max(1, (os.cpu_count() or 4) // n_proc))

print(f'Seeds {SEEDS} | GPU {N_GPU} | tiến trình {n_proc} | nhóm {groups}', flush=True)

# 1. Chuẩn bị dữ liệu một lần. Không ghi đệm: tiến độ MD5 và đọc ảnh hiện ngay trong log.
print('Bước 1: chuẩn bị dữ liệu', flush=True)
t_setup = time.time()
r = subprocess.run([sys.executable, '-u', SCRIPT], env=dict(common, SETUP_ONLY='1'))
print(f'Bước 1 xong sau {(time.time() - t_setup) / 60:.1f} phút, mã {r.returncode}', flush=True)
if r.returncode != 0:
    raise SystemExit('Chuẩn bị dữ liệu lỗi, dừng.')

# 2. Mỗi GPU một tiến trình
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
procs = []
for gpu, grp in enumerate(groups):
    env = dict(common, CUDA_VISIBLE_DEVICES=str(gpu), SEEDS=','.join(grp), SUMMARIZE='0')
    log = open(Path(LOG_DIR) / f'gpu{gpu}.log', 'w', encoding='utf-8')
    p = subprocess.Popen([sys.executable, '-u', SCRIPT], env=env, stdout=log, stderr=subprocess.STDOUT)
    procs.append((gpu, grp, p, log))
    print(f'GPU {gpu}: seeds {grp}, pid {p.pid}', flush=True)

# 3. Theo dõi tiến độ
t0 = time.time()
while any(p.poll() is None for _, _, p, _ in procs):
    time.sleep(60)
    for gpu, grp, p, _ in procs:
        lines = Path(LOG_DIR, f'gpu{gpu}.log').read_text(encoding='utf-8', errors='replace').splitlines()
        last = lines[-1] if lines else '(chưa có log)'
        state = 'xong' if p.poll() is not None else 'đang chạy'
        print(f'[{(time.time() - t0) / 60:5.1f} phút] GPU{gpu} {state}: {last}', flush=True)

failed = False
for gpu, grp, p, log in procs:
    log.close()
    print(f'GPU {gpu} kết thúc, mã {p.returncode}', flush=True)
    if p.returncode != 0:
        failed = True
        print(Path(LOG_DIR, f'gpu{gpu}.log').read_text(encoding='utf-8', errors='replace')[-2000:])

# 4. Tổng hợp
r = subprocess.run([sys.executable, '-u', SCRIPT], env=dict(common, SUMMARIZE_ONLY='1'),
                   capture_output=True, text=True, encoding='utf-8', errors='replace')
print(r.stdout[-4000:])
if failed:
    print('Có tiến trình lỗi. Xem log ở', LOG_DIR)
    raise SystemExit('Có tiến trình lỗi, mã thoát 1. Kết quả tổng hợp ở trên có thể thiếu seed.')
