// Tạo notebook Colab từ ood_calibration.py (tách theo dấu "# %% ").
// Chạy: node code/make_notebook.js (từ thư mục ma21_ood_calibration_bcn20000)
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'code', 'ood_calibration.py'), 'utf8');
const parts = src.split(/^(?=# %% )/m);
const sections = parts.filter((p) => p.startsWith('# %% '));

const toLines = (text) => {
  const lines = text.replace(/\s+$/, '').split('\n');
  return lines.map((l, i) => l + (i < lines.length - 1 ? '\n' : ''));
};
const codeCell = (text) => ({ cell_type: 'code', execution_count: null, metadata: {}, outputs: [], source: toLines(text) });
const mdCell = (text) => ({ cell_type: 'markdown', metadata: {}, source: toLines(text.trim()) });

const MD5 = '2293c9c654a3f76d6506a57ad17cddd1';

const intro = `# Mã 21: phát hiện OOD và hiệu chỉnh độ tin cậy trên BCN20000 (Colab + Google Drive)

**Trước khi chạy:**
1. Mở notebook trên Google Colab.
2. **Runtime → Change runtime type → GPU T4.**
3. Chạy cell 0. Colab sẽ hỏi quyền truy cập Google Drive: bạn tự đăng nhập và cho phép.
4. **Runtime → Run all.**

**Lưu trên Google Drive** (thư mục \`MyDrive/ma21_bcn20000\`):
- \`data/\`: tệp zip train (6,37 GB), CSV, và file ghi nhận đã kiểm MD5 ${MD5}.
- \`checkpoints/seedN/\`: \`last.pt\` (lưu sau mỗi epoch) và \`best_model.pt\`.
- \`outputs/\`: results_seed*.json, predictions_seed*.csv và summary_per_seed.csv.

Cần khoảng 6,5 GB dung lượng Drive. Kiểm tra trong Drive, mục Storage.

**Trên ổ của máy ảo** (tạm, tạo lại mỗi phiên): ảnh đã giải nén và cache ảnh 256 px. Mỗi phiên mới mất vài phút để giải nén và tạo cache.

**Chạy tiếp khi bị ngắt:** chạy lại notebook. Seed đã xong được bỏ qua, seed đang dở sẽ tiếp tục từ epoch đã lưu gần nhất.

**Lưu ý:** Colab miễn phí thường chỉ có một GPU. Phiên có thể tối đa khoảng 12 giờ. HOLDOUT mặc định là AK (chưa chốt, xem STATUS.md). Tệp test không dùng vì không có nhãn.`;

const cell0 = `# Cell 0: gắn Google Drive (Colab) và ghi đè cấu hình tùy chọn
import os
if os.path.isdir('/content'):
    from google.colab import drive
    drive.mount('/content/drive')
# os.environ["SEEDS"] = "0,1,2"
# os.environ["EPOCHS"] = "12"
# os.environ["HOLDOUT"] = "AK"
# os.environ["BATCH"] = "32"   # batch mỗi GPU
# os.environ["CACHE"] = "1"    # 1: thu nhỏ ảnh một lần (khuyến nghị)
# os.environ["RESUME"] = "1"   # 1: tiếp tục từ checkpoint và bỏ qua seed đã xong`;

const cells = [mdCell(intro), codeCell(cell0)];
for (const s of sections) cells.push(codeCell(s));

const nb = {
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    kernelspec: { name: 'python3', display_name: 'Python 3', language: 'python' },
    language_info: { name: 'python' },
  },
  cells,
};

const out = path.join(root, 'notebook', 'ood_calibration_colab.ipynb');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(nb, null, 1), 'utf8');

const check = JSON.parse(fs.readFileSync(out, 'utf8'));
console.log('cells:', check.cells.length, check.cells.map((c) => c.cell_type[0] + ':' + c.source.length).join(' '));
console.log('md5 in intro:', check.cells[0].source.join('').includes(MD5));
