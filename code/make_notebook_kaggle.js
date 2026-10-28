// Tạo notebook Kaggle: ghi file script rồi chạy song song mỗi seed trên một GPU.
// Chạy: node code/make_notebook_kaggle.js (từ thư mục ma21_ood_calibration_bcn20000)
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(root, 'code', 'ood_calibration.py'), 'utf8');
const launcher = fs.readFileSync(path.join(root, 'code', 'launch_parallel.py'), 'utf8');

const toLines = (text) => {
  const lines = text.replace(/\s+$/, '').split('\n');
  return lines.map((l, i) => l + (i < lines.length - 1 ? '\n' : ''));
};
const codeCell = (text) => ({ cell_type: 'code', execution_count: null, metadata: {}, outputs: [], source: toLines(text) });
const mdCell = (text) => ({ cell_type: 'markdown', metadata: {}, source: toLines(text.trim()) });

const MD5 = '2293c9c654a3f76d6506a57ad17cddd1';
// Hai chế độ: 'train' (huấn luyện v4 từ đầu) và 'recompute' (chấm lại checkpoint v4 vào v4r, không huấn luyện)
const MODE = process.argv[2] || 'train';
// Cấu hình các đợt xám/màu (đăng ký trước: docs mục 12, 13, 14, 16). lr = null nghĩa là giữ LR mặc định 3e-4.
// aug, aug_c, sens chỉ có ở các đợt mục 14 và 16: độ sáng, tương phản (0 = tắt) và đo độ nhạy độ sáng D1.
// Mục 16: res = RES_MATCH (E1), rot = AUG_ROT theo độ (E2), dataset = DATASET (E3). tag là đuôi thư mục output.
const EXP = {
  gray: { tag: 'v5g', grayscale: '1', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null },
  lr: { tag: 'v5h', grayscale: '1', seeds: '0,1,2,3,4,5,6,7,8,9', lr: '1e-4' },
  lrcolor: { tag: 'v3h', grayscale: '0', seeds: '0,1,2,3,4', lr: '1e-4' },
  jit: { tag: 'v3j', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, aug: '0.4', aug_c: '0.3', sens: '1' },
  jitbase: { tag: 'v3b', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, aug: '0', aug_c: '0', sens: '1' },
  res: { tag: 'v3r', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, aug: '0', aug_c: '0', sens: '1', res: '1' },
  rot: { tag: 'v3p', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, aug: '0', aug_c: '0', sens: '1', rot: '30' },
  padj: { tag: 'v3pj', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, dataset: 'pad', aug: '0.4', aug_c: '0.3', sens: '1' },
  padb: { tag: 'v3pb', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, dataset: 'pad', aug: '0', aug_c: '0', sens: '1' },
  jit2: { tag: 'v3j2', grayscale: '0', seeds: '10,11,12,13,14,15,16,17,18,19', lr: null, aug: '0.4', aug_c: '0.3', sens: '1' },
  jitbase2: { tag: 'v3b2', grayscale: '0', seeds: '10,11,12,13,14,15,16,17,18,19', lr: null, aug: '0', aug_c: '0', sens: '1' },
  // E1 chạy lại sau lỗi NaN ở đối chứng xám (fp16): cùng cấu hình res, đối chứng xám chạy fp32 (mã sửa 2026-10-21)
  res2: { tag: 'v3r2', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, aug: '0', aug_c: '0', sens: '1', res: '1' },
  // E4 (mục 17): ConvNeXt-Tiny, cùng cặp J và B như E5. backbone mặc định của các mục khác là effb0.
  e4j: { tag: 'e4j', backbone: 'convnext_t', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, aug: '0.4', aug_c: '0.3', sens: '1' },
  e4b: { tag: 'e4b', backbone: 'convnext_t', grayscale: '0', seeds: '0,1,2,3,4,5,6,7,8,9', lr: null, aug: '0', aug_c: '0', sens: '1' },
};
const EXP_MODES = Object.keys(EXP);
const RUN_TAG = MODE === 'recompute' ? 'v4r' : (EXP[MODE] ? EXP[MODE].tag : 'v4');

const trainIntro = `# Mã 21: phát hiện OOD và hiệu chỉnh độ tin cậy trên BCN20000 (Kaggle, ${RUN_TAG})

**Cài đặt notebook (bắt buộc):**
1. **Settings → Accelerator: GPU T4 x2.** Nếu để None, phiên chạy CPU và launcher dừng ngay với thông báo KHÔNG THẤY GPU.
2. **Settings → Internet: On** (trọng số ImageNet).

**Cách chạy:**
- Mỗi seed chạy trên một GPU riêng, các seed chạy song song. Mặc định SEEDS=0,1,2,3,4: GPU 0 chạy seed 0, 2, 4 lần lượt; GPU 1 chạy seed 1, 3.
- Bấm **Save & Run All (Commit)**. Chỉ output của phiên bản đã commit mới được lưu để dùng lần sau.
- Lần chạy tiếp: **Add Input** chọn output của lần commit trước (có thư mục \`ma21_bcn20000/${RUN_TAG}/checkpoints\`). Code tự khôi phục checkpoint của đúng ${RUN_TAG}, seed đã xong được bỏ qua.
- Kiểm tra trong log: dòng "Đã khôi phục từ ..." là đúng. Nếu thấy "Không có checkpoint ..." thì Add Input chưa đúng; dòng "Mục trong /kaggle/input" cho biết đang gắn gì.

**Chia dữ liệu ${RUN_TAG}:** chia theo tổn thương, giữ tỉ lệ lớp (test ~20%, val ~10% mỗi lớp). Cách chia giống v2 và v3 (cùng seed, cùng tập). Kết quả không so sánh trực tiếp với v1 (chia cũ).

**Khác v3 ở ${RUN_TAG}:** backbone ConvNeXt-Tiny (v3: EfficientNet-B0), giữ EPOCHS=30, PATIENCE=5. Thêm điểm OOD dựa trên đặc trưng (Mahalanobis, kNN) và đối chứng far-OOD thang xám. Nên so sánh v4 với v3 trên cùng seed để tách hiệu ứng backbone.
- LABEL_SMOOTH giữ bằng 0. Mukhoti và cộng sự 2020 cho thấy sau temperature scaling, làm mượt nhãn cho ECE cao hơn cross-entropy ở 11/11 cấu hình.
- Chọn siêu tham số (T, k của kNN) chỉ trên tập validation hoặc cố định trước, không chọn trên tập OOD báo cáo (Huang và cộng sự 2023).

**Lưu trong Output** (\`/kaggle/working/ma21_bcn20000/${RUN_TAG}\`):
- \`checkpoints/seedN/\`: \`last.pt\` (sau mỗi epoch) và \`best_model.pt\`.
- \`outputs/\`: results_seed*.json, predictions_seed*.csv (xác suất từng lớp, điểm OOD, mỗi ảnh một dòng), summary_per_seed.csv.
- \`outputs/seedN/\`: splits.csv, class_counts.csv, history.csv, temperature.json, reliability_*.csv, confusion_*.csv, per_class_*.csv, logits.npz, bootstrap_*.csv, run_info.json, code_snapshot_ood_calibration.py.
- \`logs/gpuN.log\`: nhật ký đầy đủ của từng tiến trình.

**Không lưu trong Output:** ảnh đã giải nén và cache. Hai thứ này nằm ở \`/tmp\`, tạo lại mỗi phiên trong vài phút.

**Dữ liệu (Add Input, không tải lại):**
- Dataset chứa zip train \`BCN_20k_train.zip\` (6,37 GB, MD5 ${MD5}) hoặc thư mục ảnh train đã giải nén.
- Dataset chứa CSV \`bcn_20k_train.csv\`. Có thể là cùng Dataset với zip.
- Code tự tìm trong Input, kể cả thư mục lồng sâu. Thiếu tệp nào thì dừng ngay và liệt kê những gì đang có, không tải từ figshare.

**Đưa output sang Google Drive (tùy chọn):** tab Output → Download, rồi upload lên Drive.

**Lưu ý:** HOLDOUT mặc định là AK (chưa chốt, xem STATUS.md). Tệp test không dùng vì không có nhãn. Phiên Kaggle tối đa khoảng 12 giờ.`;

const recompIntro = `# Mã 21: chấm lại v4 với đối chứng khớp mức thấp (Kaggle, ${RUN_TAG})

**Mục đích:** không huấn luyện lại. Lấy checkpoint v4 (ConvNeXt-Tiny, 5 seed) và chấm lại với đối chứng khớp mức thấp (cân trọng số ID theo độ bão hòa và độ dốc), cùng Mahalanobis và kNN.

**Cài đặt notebook (bắt buộc):**
1. **Settings → Accelerator: GPU T4 x2.** Nếu để None, launcher dừng ngay.
2. **Settings → Internet: On.**
3. **Add Input:** chọn output v4 đã commit (phải có \`ma21_bcn20000/v4/checkpoints\`).
4. Bấm **Save & Run All (Commit)**. Output lưu ở \`/kaggle/working/ma21_bcn20000/${RUN_TAG}\`.

**Kiểm tra trong log:** dòng "Đã khôi phục từ ..." và "đã có best_model.pt, bỏ qua huấn luyện" cho cả 5 seed. Nếu thấy "Không có checkpoint v4 đầu vào" thì Add Input chưa đúng.

**Dữ liệu:** như notebook huấn luyện: Add Input Dataset chứa zip train và Dataset chứa CSV. Không tải lại từ figshare.`;

const grayIntro = `# Mã 21: mô hình huấn luyện trên ảnh xám (Kaggle, ${RUN_TAG})

**Mục đích:** tách "phát hiện far-OOD cần màu" khỏi "bỏ màu làm đổi điểm". Toàn bộ ảnh train, test ID và far đổi sang thang xám (luminance, ba kênh). Thiết kế và quy tắc đặt trước: docs/dose_response_design.md mục 12.

**Cài đặt notebook (bắt buộc):**
1. **Settings → Accelerator: GPU T4 x2.** Nếu để None, launcher dừng ngay.
2. **Settings → Internet: On** (trọng số ImageNet).
3. **Add Input, hai Dataset (không tải lại từ figshare):**
   - Dataset chứa zip train \`BCN_20k_train.zip\` (6,37 GB, MD5 ${MD5}) hoặc thư mục ảnh train đã giải nén. Ví dụ: bcn20000.
   - Dataset chứa \`bcn_20k_train.csv\` và 5 tệp far con (bloodmnist_224.npz, pathmnist_224.npz, retinamnist_224.npz, pneumoniamnist_224.npz, breastmnist_224.npz). Ví dụ: bcn20000-farsub.
   Code tự tìm trong /kaggle/input, kể cả thư mục lồng sâu. Thiếu tệp nào thì dừng và liệt kê những gì đang có.
4. Bấm **Save & Run All (Commit)**. Output: \`/kaggle/working/ma21_bcn20000/${RUN_TAG}\`.

**Kiểm tra trong log:** dòng "CSV từ Input" và "Zip từ Input" (hoặc "Ảnh train là thư mục đã giải nén"), dòng "MD5 khớp sau ... s", dòng "Bước 1 xong sau ... phút", dòng "far-OOD: [...]" liệt kê đủ 5 đường dẫn, dòng "device: cuda", và "ID test (sau TS)" với bal_acc. Nếu thiếu tệp far, cell 0 báo lỗi rõ ràng.

**Cấu hình:** EfficientNet-B0, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=1. Tập far con tạo bởi code/make_farsub.py.`;

const lrIntro = `# Mã 21: mô hình xám với learning rate 1e-4 (Kaggle, v5h)

**Mục đích:** kiểm tra BA thấp của mô hình xám có phải do lịch huấn luyện không. Giống v5g ở mọi thứ, trừ LR = 1e-4 (v5g dùng 3e-4). Đăng ký trước: docs/dose_response_design.md mục 13. Chạy cùng đợt với v3h (đối chứng màu, LR 1e-4).

**Cài đặt notebook (bắt buộc):**
1. Settings → Accelerator: GPU T4 x2. Nếu để None, launcher dừng ngay.
2. Settings → Internet: On (trọng số ImageNet).
3. Add Input: Dataset chứa ảnh train đã giải nén và CSV (ví dụ BCN-ODD), và Dataset chứa 5 tệp far con (ví dụ bcn20000-farsub). Code tự tìm trong /kaggle/input.
4. Bấm Save & Run All (Commit). Output: /kaggle/working/ma21_bcn20000/v5h.

**Kiểm tra trong log:** dòng "CSV từ Input", "Ảnh train là thư mục đã giải nén", "far-OOD: [...]" liệt kê đủ 5 đường dẫn, "device: cuda", và 10 seed hoàn thành.

**Cấu hình:** EfficientNet-B0, LR=1e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=1.`;

const lrcolorIntro = `# Mã 21: đối chứng màu với learning rate 1e-4 (Kaggle, v3h)

**Mục đích:** mô hình màu cùng LR 1e-4 để làm BA và AUROC tham chiếu cho v5h. Đăng ký trước: docs/dose_response_design.md mục 13. Chỉ khác v5h ở GRAYSCALE = 0 và 5 seed.

**Cài đặt notebook (bắt buộc):**
1. Settings → Accelerator: GPU T4 x2.
2. Settings → Internet: On (trọng số ImageNet).
3. Add Input: cùng hai Dataset như v5h (ảnh train đã giải nén kèm CSV, và 5 tệp far con).
4. Bấm Save & Run All (Commit). Output: /kaggle/working/ma21_bcn20000/v3h.

**Kiểm tra trong log:** như v5h, nhưng ảnh giữ màu (không có dòng "xám") và có 5 seed.

**Cấu hình:** EfficientNet-B0, LR=1e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 5 seed (0–4), GRAYSCALE=0.`;

const jitIntro = `# Mã 21, A3: can thiệp bất biến độ sáng và tương phản (Kaggle, v3j)

**Mục đích:** mô hình màu giống v3, thêm tăng cường độ sáng ×U(0,6; 1,4) và tương phản ×U(0,7; 1,3) khi huấn luyện. Đối chứng là v3b (cùng cấu hình, không tăng cường). Đăng ký trước: docs/dose_response_design.md mục 14 (phiên bản 2).

**Cài đặt notebook (bắt buộc):**
1. Settings → Accelerator: GPU T4 x2.
2. Settings → Internet: On (trọng số ImageNet).
3. Add Input: Dataset chứa ảnh train đã giải nén kèm CSV (ví dụ BCN-ODD), và Dataset chứa 5 tệp far con (ví dụ bcn20000-farsub).
4. Bấm Save & Run All (Commit). Output: /kaggle/working/ma21_bcn20000/v3j.

**Kiểm tra trong log:** dòng "độ nhạy độ sáng: D1 = ..." cho mỗi seed, và 10 seed hoàn thành. Cờ AUG_BRIGHT = 0,4 nằm trong config của results_seed*.json (log không in cờ này). Nếu thiếu dòng D1, dừng và báo lại.

**Cấu hình:** EfficientNet-B0, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, AUG_BRIGHT=0,4, AUG_CONTRAST=0,3, BRIGHT_SENS=1.`;

const jitbaseIntro = `# Mã 21, A3: đối chứng không tăng cường (Kaggle, v3b)

**Mục đích:** đối chứng của v3j, cùng mọi cấu hình, không tăng cường độ sáng hay tương phản. Đăng ký trước: docs/dose_response_design.md mục 14 (phiên bản 2). Chạy song song với v3j nếu hàng đợi cho phép.

**Cài đặt notebook (bắt buộc):** như v3j (GPU T4 x2, Internet On, cùng hai Dataset). Output: /kaggle/working/ma21_bcn20000/v3b.

**Kiểm tra trong log:** dòng "độ nhạy độ sáng: D1 = ..." cho mỗi seed, và 10 seed hoàn thành. Cờ AUG_BRIGHT = 0 nằm trong config của results_seed*.json.

**Cấu hình:** EfficientNet-B0, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, AUG=0, BRIGHT_SENS=1.`;

const resIntro = `# Mã 21, E1: kiểm soát độ phân giải (Kaggle, v3r)

**Mục đích:** B_R giống B (v3b) ở mọi thứ, trừ ảnh BCN (ID train, val, test và holdout AK) đi qua 224 → 256 bilinear trước khi chuẩn hóa, để khớp với far-OOD. Far không đổi. Đây là biến đổi duy nhất. Đăng ký trước: docs/dose_response_design.md mục 16.

**Cài đặt notebook (bắt buộc):**
1. Settings → Accelerator: GPU T4 x2. Nếu để None, launcher dừng ngay.
2. Settings → Internet: On (trọng số ImageNet).
3. Add Input: Dataset chứa ảnh train đã giải nén kèm CSV (ví dụ BCN-ODD), và Dataset chứa 5 tệp far con (ví dụ bcn20000-farsub).
4. Bấm Save & Run All (Commit). Output: /kaggle/working/ma21_bcn20000/v3r.

**Kiểm tra trong log:** "device: cuda", "ID test (sau TS)" với bal_acc, và 10 seed hoàn thành. Cờ RES_MATCH nằm trong config của results_seed*.json, phải là true.

**Cấu hình:** EfficientNet-B0, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, AUG=0, BRIGHT_SENS=1, RES_MATCH=1.`;

const rotIntro = `# Mã 21, E2: placebo hình học (Kaggle, v3p)

**Mục đích:** P giống B (v3b), thêm xoay ngẫu nhiên θ ~ U(−30°, 30°) khi huấn luyện, phản chiếu biên, không đổi độ sáng và tương phản. Đối chứng để kiểm tính đặc hiệu của can thiệp độ sáng (v3j). Đăng ký trước: docs/dose_response_design.md mục 16.

**Cài đặt notebook (bắt buộc):** như v3r (GPU T4 x2, Internet On, cùng hai Dataset). Output: /kaggle/working/ma21_bcn20000/v3p.

**Kiểm tra trong log:** "device: cuda", "ID test (sau TS)" với bal_acc, và 10 seed hoàn thành. Cờ AUG_ROT nằm trong config của results_seed*.json, phải bằng 30.

**Cấu hình:** EfficientNet-B0, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, AUG_BRIGHT=0, AUG_CONTRAST=0, AUG_ROT=30, BRIGHT_SENS=1.`;

const padjIntro = `# Mã 21, E3: tập ID thứ hai PAD-UFES-20, có tăng cường (Kaggle, v3pj)

**Mục đích:** kiểm tính tổng quát của can thiệp độ sáng và tương phản trên tập ID khác: PAD-UFES-20 (ảnh chụp bằng điện thoại, 6 lớp, 2.298 ảnh). Cùng cấu hình v3j, chỉ khác dữ liệu ID. Lớp giữ lại là AK. Đối chứng: v3pb. Đăng ký trước: docs/dose_response_design.md mục 16.

**Cài đặt notebook (bắt buộc):**
1. Settings → Accelerator: GPU T4 x2. Nếu để None, launcher dừng ngay.
2. Settings → Internet: On (trọng số ImageNet).
3. Add Input (không cần BCN20000): Dataset PAD-UFES-20 chứa metadata.csv và thư mục imgs_part_1..3 (ví dụ mahdavi1202/skin-cancer), và Dataset chứa 5 tệp far con (ví dụ bcn20000-farsub).
4. Bấm Save & Run All (Commit). Output: /kaggle/working/ma21_bcn20000/v3pj.

**Kiểm tra trong log:** "far-OOD: [...]" liệt kê đủ 5 đường dẫn, "device: cuda", "ID test (sau TS)" với bal_acc, và 10 seed hoàn thành. Nếu thiếu metadata.csv hoặc imgs_part, cell báo lỗi và dừng. MEL chỉ có 52 ảnh, nên chỉ báo cáo theo lớp.

**Cấu hình:** EfficientNet-B0, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, DATASET=pad, AUG_BRIGHT=0,4, AUG_CONTRAST=0,3, BRIGHT_SENS=1.`;

const padbIntro = `# Mã 21, E3: đối chứng không tăng cường trên PAD-UFES-20 (Kaggle, v3pb)

**Mục đích:** đối chứng của v3pj trên cùng dữ liệu PAD-UFES-20, cùng mọi cấu hình, không tăng cường độ sáng hay tương phản. Đăng ký trước: docs/dose_response_design.md mục 16.

**Cài đặt notebook (bắt buộc):** như v3pj (GPU T4 x2, Internet On, cùng hai Dataset, không cần BCN20000). Output: /kaggle/working/ma21_bcn20000/v3pb.

**Kiểm tra trong log:** như v3pj. Cờ AUG_BRIGHT và AUG_CONTRAST trong config của results_seed*.json phải bằng 0.

**Cấu hình:** EfficientNet-B0, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, DATASET=pad, AUG=0, BRIGHT_SENS=1.`;

const jit2Intro = `# Mã 21, E5: thêm 10 seed cho v3j, seed 10–19 (Kaggle, v3j2)

**Mục đích:** mở rộng v3j từ 10 lên 20 seed để thu hẹp khoảng tin cậy (mục 16, E5). Cùng cấu hình và cùng dữ liệu như v3j, chỉ khác seed. Ghép với kết quả v3j để có 20 seed.

**Cài đặt notebook (bắt buộc):** như v3j (GPU T4 x2, Internet On, cùng hai Dataset). Output: /kaggle/working/ma21_bcn20000/v3j2.

**Kiểm tra trong log:** 10 seed (10–19) hoàn thành, mỗi GPU 5 seed. Không có seed nào bị bỏ qua vì đã có kết quả.

**Cấu hình:** như v3j, SEEDS=10–19.`;

const jitbase2Intro = `# Mã 21, E5: thêm 10 seed cho v3b, seed 10–19 (Kaggle, v3b2)

**Mục đích:** mở rộng đối chứng v3b từ 10 lên 20 seed (mục 16, E5). Cùng cấu hình và dữ liệu như v3b, chỉ khác seed. Ghép với kết quả v3b để có 20 seed.

**Cài đặt notebook (bắt buộc):** như v3b. Output: /kaggle/working/ma21_bcn20000/v3b2.

**Kiểm tra trong log:** 10 seed (10–19) hoàn thành, mỗi GPU 5 seed.

**Cấu hình:** như v3b, SEEDS=10–19.`;

const res2Intro = `# Mã 21, E1 chạy lại (Kaggle, v3r2): sửa lỗi NaN ở đối chứng xám

**Vì sao chạy lại:** lần v3r (ma21-E1-res-v3r) lỗi ở seed 1 khi tính AUROC đối chứng xám: NaN ở logits xám, nghi tràn số fp16. Seed 0, 2, 4, 6, 8 của v3r đã xong nhưng dùng fp16 cho đối chứng xám. Để các seed đồng nhất, chạy lại toàn bộ 10 seed với đối chứng xám ở fp32. Nhánh màu vẫn fp16. Thiết kế đăng ký: docs/dose_response_design.md mục 16, không đổi.

**Cài đặt notebook (bắt buộc):**
1. Settings → Accelerator: GPU T4 x2. Settings → Internet: On.
2. Add Input: Dataset BCN-ODD (ảnh train giải nén kèm CSV), Dataset bcn20000-farsub (5 tệp far con), và output của notebook ma21-E1-res-v3r (để lấy checkpoint seed 1 cho chẩn đoán).
3. Bấm Save & Run All (Commit). Output: /kaggle/working/ma21_bcn20000/v3r2.

**Kiểm tra trong log:** dòng "chẩn đoán" (fp16 và fp32, ảnh far màu và xám). Dòng "Chẩn đoán xong: fp32 không có giá trị lỗi" phải xuất hiện. Nếu fp32 vẫn có giá trị lỗi, notebook dừng trước khi huấn luyện. Sau đó: "device: cuda", "ID test (sau TS)" với bal_acc, và 10 seed hoàn thành.

**Cấu hình:** EfficientNet-B0, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, AUG=0, BRIGHT_SENS=1, RES_MATCH=1, đối chứng xám fp32.`;

const e4jIntro = `# Mã 21, E4: ConvNeXt-T, nhánh J (Kaggle, e4j)

**Mục đích:** E4 (docs/dose_response_design.md mục 17): kiến trúc ConvNeXt-Tiny với cùng cấu hình tăng cường của J (v3j). Đăng ký trước khi chạy, 2026-10-27.

**Cài đặt notebook (bắt buộc):** như v3j (GPU T4 x2, Internet On, cùng hai Dataset: BCN-ODD và bcn20000-farsub). Output: /kaggle/working/ma21_bcn20000/e4j.

**Kiểm tra trong log:** dòng "device: cuda", dòng "độ nhạy độ sáng: D1 = ..." cho mỗi seed, và 10 seed hoàn thành. Cờ BACKBONE = convnext_t nằm trong config của results_seed*.json. Nếu seed nào lỗi, dừng và báo lại; không thay bằng seed khác.

**Cấu hình:** ConvNeXt-Tiny, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, AUG_BRIGHT=0,4, AUG_CONTRAST=0,3, BRIGHT_SENS=1.`;

const e4bIntro = `# Mã 21, E4: ConvNeXt-T, nhánh B (Kaggle, e4b)

**Mục đích:** đối chứng của e4j: ConvNeXt-Tiny, không tăng cường độ sáng hay tương phản. Đăng ký trước khi chạy, 2026-10-27 (docs/dose_response_design.md mục 17).

**Cài đặt notebook (bắt buộc):** như e4j (GPU T4 x2, Internet On, cùng hai Dataset). Output: /kaggle/working/ma21_bcn20000/e4b.

**Kiểm tra trong log:** dòng "device: cuda", dòng "độ nhạy độ sáng: D1 = ..." cho mỗi seed, và 10 seed hoàn thành. Cờ BACKBONE = convnext_t nằm trong config của results_seed*.json. Nếu seed nào lỗi, dừng và báo lại; không thay bằng seed khác.

**Cấu hình:** ConvNeXt-Tiny, LR 3e-4, EPOCHS=30, PATIENCE=5, HOLDOUT=AK, 10 seed (0–9), GRAYSCALE=0, AUG=0, BRIGHT_SENS=1.`;

const INTROS = {
  recompute: recompIntro, gray: grayIntro, lr: lrIntro, lrcolor: lrcolorIntro, jit: jitIntro, jitbase: jitbaseIntro,
  res: resIntro, rot: rotIntro, padj: padjIntro, padb: padbIntro, jit2: jit2Intro, jitbase2: jitbase2Intro, res2: res2Intro, e4j: e4jIntro, e4b: e4bIntro,
};
const intro = INTROS[MODE] || trainIntro;

const cell0 = `# Cell 0: cấu hình (bỏ dấu # để đổi)
import os
os.environ.setdefault("SEEDS", "0,1,2,3,4")    # mỗi seed một GPU; GPU chạy tuần tự nếu seed nhiều hơn GPU
os.environ.setdefault("RUN_TAG", "${RUN_TAG}")
os.environ.setdefault("EPOCHS", "30")          # v3: tăng từ 12 (lịch cosine trải trên 30 epoch)
os.environ.setdefault("PATIENCE", "5")         # v3: tăng từ 3, để các epoch thêm có cơ hội cải thiện
os.environ.setdefault("BACKBONE", "convnext_t") # v4: ConvNeXt-Tiny; đặt "effb0" để khớp v3
os.environ.setdefault("FAR_GRAY", "1")          # đối chứng far-OOD: đổi ảnh có màu sang thang xám rồi chấm lại${MODE === 'recompute' ? '\nos.environ.setdefault("RESTORE_TAG", "v4")  # lấy checkpoint v4 từ Input\nos.environ.setdefault("RECOMPUTE", "1")       # chấm lại từ best_model.pt, không huấn luyện' : ''}
# os.environ["HOLDOUT"] = "AK"
# os.environ["BATCH"] = "64"               # batch mỗi GPU
# os.environ["RESUME"] = "1"               # tiếp tục từ checkpoint, bỏ qua seed đã xong
os.environ.setdefault("FAR_ENABLE", "1")   # far-OOD: tải ba bộ MedMNIST+ 224 px (CC BY 4.0) từ Zenodo, ~373 MB${EXP_MODES.includes(MODE) ? `
# ${RUN_TAG}: ${EXP[MODE].backbone === 'convnext_t' ? 'ConvNeXt-Tiny (E4, mục 17), far lấy từ Dataset gắn làm Input (không tải Zenodo)' : 'EfficientNet-B0 như v3, far lấy từ Dataset gắn làm Input (không tải Zenodo)'}
os.environ["BACKBONE"] = "${EXP[MODE].backbone || 'effb0'}"
os.environ["SEEDS"] = "${EXP[MODE].seeds}"
os.environ["GRAYSCALE"] = "${EXP[MODE].grayscale}"${EXP[MODE].lr ? `
os.environ["LR"] = "${EXP[MODE].lr}"          # đăng ký trước, docs mục 13` : ''}${EXP[MODE].aug !== undefined ? `
os.environ["EPOCHS"] = "30"                   # đăng ký trước, docs mục 14
os.environ["PATIENCE"] = "5"
os.environ["AUG_BRIGHT"] = "${EXP[MODE].aug}"      # độ sáng b ~ U(1−a, 1+a); 0 = đối chứng B
os.environ["AUG_CONTRAST"] = "${EXP[MODE].aug_c}"  # tương phản c ~ U(1−a, 1+a); 0 = đối chứng B
os.environ["BRIGHT_SENS"] = "${EXP[MODE].sens}"    # đo D1 trên ID test${EXP[MODE].res ? `
os.environ["RES_MATCH"] = "${EXP[MODE].res}"       # E1 (mục 16): ảnh ID qua 224 -> 256 bilinear trước khi cắt` : ''}${EXP[MODE].rot ? `
os.environ["AUG_ROT"] = "${EXP[MODE].rot}"         # E2 (mục 16): xoay ngẫu nhiên ±${EXP[MODE].rot} độ, không đổi độ sáng` : ''}${EXP[MODE].dataset ? `
os.environ["DATASET"] = "${EXP[MODE].dataset}"     # E3 (mục 16): PAD-UFES-20 làm ID, không dùng BCN20000` : ''}` : ''}
os.environ["FAR_ENABLE"] = "0"
import glob
_far_names = ["retinamnist_224.npz", "pneumoniamnist_224.npz", "breastmnist_224.npz", "bloodmnist_224.npz", "pathmnist_224.npz"]
_far_files = []
for n in _far_names:
    hits = sorted(glob.glob("/kaggle/input/**/" + n, recursive=True))  # Dataset từ giao diện nằm sâu 3 cấp
    if hits:
        _far_files.append(hits[0])
if len(_far_files) != len(_far_names):
    raise SystemExit("Thiếu tệp far-OOD trong Input: cần đủ 5 tệp " + ", ".join(_far_names))
os.environ["FAR_NPZ"] = ",".join(_far_files)
print("far-OOD:", _far_files)` : ''}`;

const cellDir = `import os
os.makedirs('/tmp/ma21_code', exist_ok=True)
print('thư mục code sẵn sàng')`;

const cellWrite = `%%writefile /tmp/ma21_code/ood_calibration.py
${script.replace(/\s+$/, '')}`;

const cellLauncher = launcher.replace(/\s+$/, '');

// Chẩn đoán NaN (mục điều tra 2026-10-21): cùng checkpoint seed 1 của v3r, cùng ảnh far; so fp16 với fp32.
// Chỉ chạy ở notebook res2. Dừng nếu fp32 vẫn có giá trị không hữu hạn, trước khi huấn luyện.
const diagCell = `# Chẩn đoán NaN (E1, seed 1): cùng checkpoint, cùng ảnh far; so fp16 với fp32 và đếm giá trị không hữu hạn
import sys, glob, numpy as np, torch
sys.path.insert(0, "/tmp/ma21_code")
import ood_calibration as oc
dev = torch.device("cuda")
cks = sorted(glob.glob("/kaggle/input/**/checkpoints/seed1/best_model.pt", recursive=True))
print("checkpoint seed1:", cks)
if not cks:
    raise SystemExit("Thiếu checkpoint seed1 của v3r trong Input (Add Input: output ma21-E1-res-v3r).")
ck = torch.load(cks[0], map_location="cpu", weights_only=False)
n_cls = [v for k, v in ck["state"].items() if k.startswith("classifier") and v.dim() == 2][0].shape[0]
model = oc.build_model(n_cls).to(dev)
model.load_state_dict(ck["state"])
Xfar, far_src = oc.build_far_array()
rows = np.arange(len(far_src))
ys = np.full(len(far_src), -1)
fp32_bad = 0
for name, X in [("màu", Xfar), ("xám", oc.to_gray_batch(Xfar, rows))]:
    for use_amp in (True, False):
        f, L, _ = oc.get_feats_logits(model, X, rows, ys, dev, use_amp)
        bf = ~torch.isfinite(f).all(dim=1)
        bl = ~torch.isfinite(L).all(dim=1)
        if (not use_amp) and name == "xám":
            fp32_bad = int(bl.sum()) + int(bf.sum())
        print("chẩn đoán", name, "fp16" if use_amp else "fp32", "| đặc trưng lỗi", int(bf.sum()), "| logits lỗi", int(bl.sum()), "/", len(L))
        for src in np.unique(far_src):
            m = torch.as_tensor(far_src == src)
            print("    ", src, "logits lỗi", int((bl & m).sum()), "/", int(m.sum()))
if fp32_bad:
    raise RuntimeError("fp32 vẫn có giá trị không hữu hạn trên ảnh far xám: dừng, không huấn luyện.")
print("Chẩn đoán xong: fp32 không có giá trị lỗi.")`;

const cells = [
  mdCell(intro),
  codeCell(cell0),
  codeCell(cellDir),
  codeCell(cellWrite),
  ...(MODE === 'res2' ? [codeCell(diagCell)] : []),
  codeCell(cellLauncher),
];

const nb = {
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    kernelspec: { name: 'python3', display_name: 'Python 3', language: 'python' },
    language_info: { name: 'python' },
  },
  cells,
};

const OUT_NAMES = {
  recompute: 'ood_recompute_v4_kaggle.ipynb', gray: 'ood_gray_v5g_kaggle.ipynb', lr: 'ood_lr_v5h_kaggle.ipynb', lrcolor: 'ood_lr_v3h_kaggle.ipynb',
  jit: 'ood_jit_v3j_kaggle.ipynb', jitbase: 'ood_jitbase_v3b_kaggle.ipynb', res: 'ood_res_v3r_kaggle.ipynb', rot: 'ood_rot_v3p_kaggle.ipynb',
  padj: 'ood_padj_v3pj_kaggle.ipynb', padb: 'ood_padb_v3pb_kaggle.ipynb', jit2: 'ood_jit2_v3j2_kaggle.ipynb', jitbase2: 'ood_jitbase2_v3b2_kaggle.ipynb', res2: 'ood_res2_v3r2_kaggle.ipynb',
  e4j: 'ood_e4j_kaggle.ipynb', e4b: 'ood_e4b_kaggle.ipynb',
};
const out = path.join(root, 'notebook', OUT_NAMES[MODE] || 'ood_calibration_kaggle.ipynb');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(nb, null, 1), 'utf8');

const check = JSON.parse(fs.readFileSync(out, 'utf8'));
console.log('cells:', check.cells.length, check.cells.map((c) => c.cell_type[0] + ':' + c.source.length).join(' '));
console.log('md5 in intro:', check.cells[0].source.join('').includes(MD5));
console.log('writefile magic first line:', check.cells[3].source[0].trim());
