# -*- coding: utf-8 -*-
"""
Sinh notebook Colab cho E2, E3, E5 từ notebook Kaggle đã kiểm chứng.

Giữ nguyên ô ghi mã (writefile) và ô launcher của notebook Kaggle, chỉ thay phần giới thiệu và ô cấu hình
bằng đường dẫn Google Drive và ổ tạm. Dùng: python code/make_notebook_colab.py padj|padb|rot|jit2|jitbase2

Đầu vào trên Drive:
  MyDrive/ma21_bcn20000/data/farsub/*.npz       (năm tệp far-OOD, dùng cho mọi arm)
  PAD (padj, padb): MyDrive/ma21_bcn20000/archive.zip (hoặc .../data/archive.zip), khoảng 3,6 GB
BCN (rot, jit2, jitbase2): không cần tải lên. Ô cấu hình tải BCN_20k_train.zip (6,37 GB) và CSV từ figshare
  về ổ tạm /content/local, zip được kiểm MD5 trong ood_calibration.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = {
    'padj': {'kaggle': 'ood_padj_v3pj_kaggle.ipynb', 'out': 'ood_colab_padj_v3pj.ipynb', 'tag': 'v3pj',
             'dataset': 'pad', 'aug_b': '0.4', 'aug_c': '0.3', 'rot': '0', 'label': 'J_pad (can thiệp độ sáng và tương phản)'},
    'padb': {'kaggle': 'ood_padb_v3pb_kaggle.ipynb', 'out': 'ood_colab_padb_v3pb.ipynb', 'tag': 'v3pb',
             'dataset': 'pad', 'aug_b': '0', 'aug_c': '0', 'rot': '0', 'label': 'B_pad (đối chứng không tăng cường)'},
    'rot': {'kaggle': 'ood_rot_v3p_kaggle.ipynb', 'out': 'ood_colab_rot_v3p.ipynb', 'tag': 'v3p',
            'dataset': 'bcn', 'seeds': '0,1,2,3,4,5,6,7,8,9', 'aug_b': '0', 'aug_c': '0', 'rot': '30',
            'title': 'E2: placebo hình học (xoay ±30°)', 'label': 'P (placebo hình học, xoay ±30°, E2)'},
    'jit2': {'kaggle': 'ood_jit2_v3j2_kaggle.ipynb', 'out': 'ood_colab_jit2_v3j2.ipynb', 'tag': 'v3j2',
             'dataset': 'bcn', 'seeds': '10,11,12,13,14,15,16,17,18,19', 'aug_b': '0.4', 'aug_c': '0.3', 'rot': '0',
             'title': 'E5: thêm seed 10–19 cho J', 'label': 'J, seed 10–19 (E5)'},
    'jitbase2': {'kaggle': 'ood_jitbase2_v3b2_kaggle.ipynb', 'out': 'ood_colab_jitbase2_v3b2.ipynb', 'tag': 'v3b2',
                 'dataset': 'bcn', 'seeds': '10,11,12,13,14,15,16,17,18,19', 'aug_b': '0', 'aug_c': '0', 'rot': '0',
                 'title': 'E5: thêm seed 10–19 cho B', 'label': 'B, seed 10–19 (E5)'},
    # Kiểm nền tảng: seed 0 của B (v3b, Kaggle) chạy lại trên Colab, cùng cấu hình, tag riêng để không ghi đè v3b
    'bpar': {'kaggle': 'ood_jitbase2_v3b2_kaggle.ipynb', 'out': 'ood_colab_bpar_v3bpar.ipynb', 'tag': 'v3bpar',
             'dataset': 'bcn', 'seeds': '0', 'aug_b': '0', 'aug_c': '0', 'rot': '0',
             'title': 'kiểm nền tảng: seed 0 của B (v3b) chạy lại',
             'intro': """# Mã 21, kiểm nền tảng: seed 0 của B (v3b) chạy lại trên Colab (v3bpar)

**Mục đích:** đo chênh lệch do nền tảng. Cùng cấu hình và seed 0 như v3b trên Kaggle; so results_seed0.json của hai bản. Chỉ dùng để kiểm tra, không đưa vào E5. Kết quả ghi đè không xảy ra: tag riêng v3bpar.

**Cài đặt:** T4 GPU, Run all, cấp quyền Drive khi được hỏi. Khoảng 10 phút (giải mã ảnh vài phút, một seed khoảng 5 phút).

**Kiểm tra:** "Seeds ['0']", "Bước 1 xong", "GPU 0 kết thúc, mã 0". File cần so: MyDrive/ma21_bcn20000/v3bpar/outputs/results_seed0.json."""},
    # Lặp lại đúng bpar để đo độ lặp lại trên Colab. Tag khác để RESUME không bỏ qua seed 0.
    'bpar2': {'kaggle': 'ood_jitbase2_v3b2_kaggle.ipynb', 'out': 'ood_colab_bpar2_v3bpar2.ipynb', 'tag': 'v3bpar2',
              'dataset': 'bcn', 'seeds': '0', 'aug_b': '0', 'aug_c': '0', 'rot': '0',
              'title': 'kiểm độ lặp lại: seed 0 của B chạy lần 2 trên Colab',
              'intro': """# Mã 21, kiểm độ lặp lại: seed 0 của B chạy lần 2 trên Colab (v3bpar2)

**Mục đích:** lặp lại đúng bpar (v3bpar, cùng cấu hình, cùng seed 0, cùng nền tảng). Nếu hai run Colab trùng khớp nhưng khác Kaggle thì chênh lệch là do nền tảng; nếu hai run Colab khác nhau thì là nhiễu chạy lại. Chỉ dùng để kiểm tra, không đưa vào E5.

**Cài đặt:** T4 GPU, Run all, cấp quyền Drive khi được hỏi. Khoảng 10 phút.

**Kiểm tra:** "Seeds ['0']", "Bước 1 xong", "GPU 0 kết thúc, mã 0". File cần so: MyDrive/ma21_bcn20000/v3bpar2/outputs/results_seed0.json."""},
    # Kiểm mã cũ: ô ghi mã và launcher lấy từ notebook Kaggle v3b (mã đã chạy, SHA 0e2feefc79fa sau chuẩn hóa).
    # Cùng cấu hình seed 0 của B, tag riêng v3bold. Chỉ dùng để tách ảnh hưởng mã và môi trường.
    'bold': {'kaggle': 'ood_jitbase_v3b_kaggle.ipynb', 'out': 'ood_colab_bold_v3bold.ipynb', 'tag': 'v3bold',
             'dataset': 'bcn', 'seeds': '0', 'aug_b': '0', 'aug_c': '0', 'rot': '0',
             'title': 'kiểm mã cũ: seed 0 của B (mã v3b) trên Colab',
             'intro': """# Mã 21, kiểm mã cũ: seed 0 của B chạy bằng mã v3b cũ, trên Colab (v3bold)

**Mục đích:** tách ảnh hưởng của mã và môi trường. Ô ghi mã và launcher lấy từ notebook Kaggle v3b (mã đã chạy ra kết quả Kaggle, trước sửa đường xám ngày 2026-10-21). Cùng seed 0, cùng cấu hình, chỉ khác nền tảng. Nếu kết quả trùng Kaggle v3b seed 0 thì chênh lệch hoàn toàn do môi trường; nếu khác thì một phần do mã. Chỉ dùng để kiểm tra, không đưa vào E5.

**Cài đặt:** T4 GPU, Run all, cấp quyền Drive khi được hỏi. Khoảng 10 phút.

**Kiểm tra:** "Seeds ['0']", "Bước 1 xong", "GPU 0 kết thúc, mã 0". File cần so: MyDrive/ma21_bcn20000/v3bold/outputs/results_seed0.json."""},
}

FAR_FILES = ['retinamnist_224.npz', 'pneumoniamnist_224.npz', 'breastmnist_224.npz', 'bloodmnist_224.npz',
             'pathmnist_224.npz']
BCN_ZIP_BYTES = 6368901290


def to_lines(text):
    text = text.strip('\n')
    lines = text.split('\n')
    return [l + ('\n' if i < len(lines) - 1 else '') for i, l in enumerate(lines)]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': to_lines(text)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': to_lines(text)}


def intro_pad(spec):
    return f"""# Mã 21, E3: {spec['label']} trên Colab (PAD-UFES-20, {spec['tag']})

**Mục đích:** chạy E3 trên Colab vì quota GPU Kaggle tuần này đã hết. Cùng cấu hình như bản Kaggle {spec['tag']}, khác chỗ đường dẫn dữ liệu. Thiết kế đăng ký: docs/dose_response_design.md mục 16.

**Chuẩn bị (làm một lần, trước khi bấm Run):**
1. Google Drive: `archive.zip` (ảnh PAD-UFES-20, 3,6 GB) đặt trong `MyDrive/ma21_bcn20000/` hoặc `MyDrive/ma21_bcn20000/data/`; thư mục `farsub/` gồm năm tệp `.npz` (340 MB) đặt trong `MyDrive/ma21_bcn20000/data/`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. Lần đầu sẽ hỏi cấp quyền truy cập Drive.

**Kiểm tra trong log:** "device: cuda", dòng "chẩn đoán" không có lỗi (nếu có ô chẩn đoán), "ID test (sau TS)" với bal_acc, và 10 seed hoàn thành. Cờ DATASET=pad, AUG_BRIGHT={spec['aug_b']} và AUG_CONTRAST={spec['aug_c']} nằm trong config của results_seed*.json.

**Kết quả lưu ở:** `MyDrive/ma21_bcn20000/{spec['tag']}/` (checkpoint, outputs, logs). Phiên Colab có thể bị ngắt; chạy lại cùng notebook sẽ tiếp tục từ seed chưa xong (RESUME).

**Lưu ý:** giấy phép PAD-UFES-20 là CC BY 4.0 theo bài mô tả dữ liệu (PMC7479321). Ghi công khi công bố."""


def intro_bcn(spec):
    return f"""# Mã 21, {spec['title']} trên Colab ({spec['tag']})

**Mục đích:** chạy trên Colab vì quota GPU Kaggle đã hết. Cùng cấu hình như bản Kaggle {spec['tag']}, khác chỗ đường dẫn dữ liệu và nơi lưu kết quả. Thiết kế đăng ký: docs/dose_response_design.md mục 16.

**Chuẩn bị (không cần tải lên dữ liệu BCN):**
1. Google Drive: thư mục `MyDrive/ma21_bcn20000/data/farsub/` gồm năm tệp `.npz` (340 MB). Đã có sẵn nếu chạy E3 trước.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. Lần đầu sẽ hỏi cấp quyền truy cập Drive.

**Thời gian ước tính:** tải BCN_20k_train.zip (6,37 GB) từ figshare vài phút; giải mã 12.413 ảnh train khoảng 20 phút (một lần cho cả phiên); mỗi seed khoảng 10–15 phút trên T4 (ước tính từ Kaggle). Tổng khoảng 2,5 giờ GPU cho 10 seed, cộng phần giải mã.

**Kiểm tra trong log:** "Kiểm MD5 zip ... MD5 khớp", "Đọc và thu nhỏ 12413 ảnh", "device: cuda", "ID test (sau TS)" với bal_acc, và 10 seed hoàn thành. Cờ {('AUG_ROT=30' if spec['rot'] != '0' else 'AUG_BRIGHT=' + spec['aug_b'] + ', AUG_CONTRAST=' + spec['aug_c'])} nằm trong config của results_seed*.json.

**Kết quả lưu ở:** `MyDrive/ma21_bcn20000/{spec['tag']}/` (checkpoint, outputs, logs). Phiên Colab có thể bị ngắt; chạy lại cùng notebook sẽ tiếp tục từ seed chưa xong (RESUME), nhưng phải giải mã lại ảnh (khoảng 20 phút)."""


def config_pad(spec):
    return f"""# Cell 0: cấu hình Colab (bỏ dấu # để đổi)
from google.colab import drive
drive.mount('/content/drive')
import os, zipfile
DRIVE = "/content/drive/MyDrive/ma21_bcn20000"
ARCHIVE = next((p for p in [f"{{DRIVE}}/data/archive.zip", f"{{DRIVE}}/archive.zip"] if os.path.exists(p)), None)
PAD_DIR = "/content/pad_ufes20"                     # giải nén ở ổ tạm mỗi phiên (nhanh hơn đọc từ Drive)
if not os.path.exists(f"{{PAD_DIR}}/metadata.csv"):
    if ARCHIVE is None:
        raise SystemExit("Thiếu archive.zip trong " + DRIVE + " (cần data/archive.zip hoặc archive.zip). Tải lên trước.")
    os.makedirs(PAD_DIR, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(PAD_DIR)
    print("đã giải nén PAD vào", PAD_DIR)
FAR_FILES = {FAR_FILES!r}
FAR_NPZ = [f"{{DRIVE}}/data/farsub/{{n}}" for n in FAR_FILES]
miss = [p for p in FAR_NPZ if not os.path.exists(p)]
if miss:
    raise SystemExit("Thiếu tệp far-OOD trên Drive: " + ", ".join(miss))
os.environ["PAD_DIR"] = PAD_DIR
os.environ["FAR_NPZ"] = ",".join(FAR_NPZ)
os.environ["REQUIRE_GPU"] = "1"                     # dừng nếu không có GPU, tránh chạy hàng giờ trên CPU
os.environ["DATASET"] = "pad"
os.environ["BACKBONE"] = "effb0"
os.environ["SEEDS"] = "0,1,2,3,4,5,6,7,8,9"
os.environ["EPOCHS"] = "30"
os.environ["PATIENCE"] = "5"
os.environ["GRAYSCALE"] = "0"
os.environ["AUG_BRIGHT"] = "{spec['aug_b']}"
os.environ["AUG_CONTRAST"] = "{spec['aug_c']}"
os.environ["AUG_ROT"] = "0"
os.environ["RES_MATCH"] = "0"
os.environ["BRIGHT_SENS"] = "1"
os.environ["FAR_ENABLE"] = "0"
os.environ["FAR_GRAY"] = "1"
os.environ["RUN_TAG"] = "{spec['tag']}"
os.environ["LOG_DIR"] = f"{{DRIVE}}/{spec['tag']}/logs"
print("cấu hình xong:", os.environ["RUN_TAG"], "| PAD:", PAD_DIR, "| far:", len(FAR_NPZ), "tệp")"""


def config_bcn(spec):
    return f"""# Cell 0: cấu hình Colab (bỏ dấu # để đổi)
from google.colab import drive
drive.mount('/content/drive')
import os, subprocess
DRIVE = "/content/drive/MyDrive/ma21_bcn20000"
LOCAL = "/content/local"                            # zip và CSV BCN ở ổ tạm (đọc ảnh nhanh hơn từ Drive)
os.makedirs(LOCAL, exist_ok=True)
ZIP_URL = "https://ndownloader.figshare.com/files/45833178"   # BCN_20k_train.zip, 6,37 GB (MD5 kiểm trong mã)
CSV_URL = "https://ndownloader.figshare.com/files/49291684"   # bcn_20k_train.csv
BCN_ZIP = f"{{LOCAL}}/BCN_20k_train.zip"
BCN_CSV = f"{{LOCAL}}/bcn_20k_train.csv"
if not os.path.exists(BCN_CSV):
    subprocess.run(["wget", "-q", "-O", BCN_CSV, CSV_URL], check=True)
if not os.path.exists(BCN_ZIP) or os.path.getsize(BCN_ZIP) != {BCN_ZIP_BYTES}:
    subprocess.run(["wget", "-q", "-c", "-O", BCN_ZIP, ZIP_URL], check=True)   # -c: tiếp tục nếu bị ngắt
FAR_FILES = {FAR_FILES!r}
FAR_NPZ = [f"{{DRIVE}}/data/farsub/{{n}}" for n in FAR_FILES]
miss = [p for p in FAR_NPZ if not os.path.exists(p)]
if miss:
    raise SystemExit("Thiếu tệp far-OOD trên Drive: " + ", ".join(miss))
os.environ["DRIVE_ROOT"] = DRIVE                    # checkpoint và outputs ghi thẳng lên Drive
os.environ["LOCAL"] = LOCAL
os.environ["DATASET"] = "bcn"
os.environ["BCN_ZIP"] = BCN_ZIP
os.environ["DATA_CSV"] = BCN_CSV
os.environ["FAR_NPZ"] = ",".join(FAR_NPZ)
os.environ["REQUIRE_GPU"] = "1"                     # dừng nếu không có GPU, tránh chạy hàng giờ trên CPU
os.environ["BACKBONE"] = "effb0"
os.environ["SEEDS"] = "{spec['seeds']}"
os.environ["EPOCHS"] = "30"
os.environ["PATIENCE"] = "5"
os.environ["GRAYSCALE"] = "0"
os.environ["AUG_BRIGHT"] = "{spec['aug_b']}"
os.environ["AUG_CONTRAST"] = "{spec['aug_c']}"
os.environ["AUG_ROT"] = "{spec['rot']}"
os.environ["RES_MATCH"] = "0"
os.environ["BRIGHT_SENS"] = "1"
os.environ["FAR_ENABLE"] = "0"
os.environ["FAR_GRAY"] = "1"
os.environ["RUN_TAG"] = "{spec['tag']}"
os.environ["LOG_DIR"] = f"{{DRIVE}}/{spec['tag']}/logs"
print("cấu hình xong:", os.environ["RUN_TAG"], "| seed:", os.environ["SEEDS"], "| zip:", BCN_ZIP, "| far:", len(FAR_NPZ), "tệp")"""


def build(arm):
    spec = ARMS[arm]
    src = json.loads((ROOT / 'notebook' / spec['kaggle']).read_text(encoding='utf-8'))
    kept = src['cells']
    # Ô 2, 3, 4 của notebook Kaggle: tạo thư mục mã, ghi tệp mã, launcher. Giữ nguyên.
    writefile_cell, launcher_cell = kept[3], kept[4]
    assert writefile_cell['source'][0].strip().startswith('%%writefile'), 'sai thứ tự ô'
    assert kept[2]['source'], 'thiếu ô tạo thư mục'
    if spec['dataset'] == 'pad':
        intro, config = intro_pad(spec), config_pad(spec)
    else:
        intro, config = spec.get('intro') or intro_bcn(spec), config_bcn(spec)

    cells = [md(intro), code(config), kept[2], writefile_cell, launcher_cell]
    out = {'nbformat': 4, 'nbformat_minor': 5,
           'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
                        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                        'language_info': {'name': 'python'}},
           'cells': [dict(c, id=f'c{i}') for i, c in enumerate(cells)]}
    path = ROOT / 'notebook' / spec['out']
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


if __name__ == '__main__':
    arm = sys.argv[1] if len(sys.argv) > 1 else 'padb'
    if arm not in ARMS:
        raise SystemExit('dùng: ' + ' | '.join(ARMS))
    print('đã ghi', build(arm))
