# Thiết kế đường đáp ứng liều: màu hướng về far so với placebo vuông góc

Phiên bản: 0.3 (2026-10-18). v0.2 đã chạy trên v3: không bộ có màu nào đạt quyết định vì placebo không cân bằng độ bão hòa. v0.3 thay placebo để khớp độ lớn chroma trung bình. Trạng thái: chờ kiểm khả thi bằng thống kê ảnh và đánh giá độc lập.

## 1. Câu hỏi

Phát hiện far-OOD của mô hình (năng lượng sau nhiệt độ T) có phụ thuộc vào hướng của thay đổi màu về phía bộ far hay không?
- (a) Khi đưa sắc độ của ảnh ID về phía sắc độ của bộ far với liều α, phát hiện có giảm theo α không?
- (b) Sự giảm có đặc hiệu theo hướng về far, hay chỉ phụ thuộc độ lớn thay đổi sắc độ?

Bối cảnh (v3, 5 seed): khớp sắc độ theo hướng far không làm giảm tín hiệu võng mạc (0,916 → 0,917) nhưng cân bằng độ bão hòa chưa đạt (SMD 0,33). Với PathMNIST, khớp làm giảm tín hiệu (0,856 → 0,738) và placebo xoay cũng giảm tương tự (0,759), nhưng placebo xoay ở v3 không cùng độ lớn và có σ đích bị hoán đổi, nên so sánh đó không kết luận được.

## 2. Nhánh và liều

Không gian Lab. Chỉ đổi kênh a, b (sắc độ); giữ nguyên L. Liều α ∈ {0,25; 0,5; 1,0}.

Ký hiệu, cho mỗi bộ far và mỗi seed:
- μ_ID, σ_ID: trung bình và độ lệch chuẩn của a, b của ảnh ID test (theo pixel, đã lấy mẫu).
- μ_far, σ_far: của bộ far (theo pixel, đã lấy mẫu).
- d = μ_far − μ_ID (vector hai chiều trên (a, b)).

Nhánh:
- `orig` (liều 0): ảnh ID gốc.
- `dir_α` (hướng về far): đích trung bình μ_dir = μ_ID + α·d; đích độ lệch chuẩn σ_dir = σ_ID + α(σ_far − σ_ID). Khớp từng kênh a, b theo (μ_dir, σ_dir); giữ L.
- `rot_α` (placebo vuông góc, v0.3): đích trung bình μ_rot = μ_ID + α·R_θ·d, đích độ lệch chuẩn σ_rot = σ_dir. Góc θ được chọn theo quy tắc sau, không theo kết quả mô hình:
  - Ràng buộc hình học: |cos(R_θ·d, d)| ≤ 0,3 (góc giữa hai độ dời nằm trong [72°, 108°]).
  - Ràng buộc độ lớn: trong các θ thỏa ràng buộc hình học, chọn θ làm độ lớn chroma trung bình (trung bình theo pixel của √(a² + b²)) của placebo gần nhất với nhánh dir cùng liều. θ được quét với bước 1°, trên mẫu 20 000 pixel của tập ID, chỉ dùng thống kê ảnh.
  - Nếu không có θ nào làm chênh lệch độ lớn chroma ≤ 2%, liều đó được đánh dấu không khả thi và không đưa vào phân tích cặp.
  Giữ L.

Tính chất (v0.3): σ đích giống nhau; |cos| ≤ 0,3 giữa độ dời trung bình của hai nhánh; độ lớn chroma trung bình của hai nhánh cách nhau ≤ 2%. Hai nhánh khác nhau chủ yếu ở hướng của độ dời. Ràng buộc độ lớn chroma được thêm vì độ bão hòa RGB phụ thuộc vào độ lớn chroma, không chỉ vào độ dời (v0.2 chỉ khớp độ dời nên độ bão hòa lệch, ví dụ SMD 1,78 với PathMNIST).

Bộ xám (breastmnist, pneumoniamnist): chỉ có `dir_α` (đích sắc độ 0, nên đây là khử màu một phần). Không có placebo, nên bộ xám không có kết luận về hướng.

Không dùng θ ngẫu nhiên: placebo là phép xoay 90° cố định, không phụ thuộc seed hay bộ far.

Không áp dụng làm mờ hoặc làm sắc nét trong thiết kế này. Độ dốc được báo cáo nhưng không dùng làm cổng.

## 3. Số liệu và kiểm tra

Với mỗi nhánh, liều và seed:
- AUROC so với far gốc (năng lượng sau T và MSP sau T). Đây là đại lượng chính.
- AUROC so với ID gốc (shift). Chỉ báo cáo, không dùng làm cổng (xem mục 6).
- SMD của độ bão hòa RGB, của L, và của độ dốc, so với far.
- ΔE76 trung bình (Lab) so với ảnh gốc, và độ lớn sắc độ trung bình.

Đường liều: hiệu theo cặp D_α = AUROC(dir_α) − AUROC(rot_α) theo từng seed, và độ dốc theo α.

## 4. Cổng và quy tắc quyết định (đặt trước khi chạy)

Cổng ΔE (kiểm tra độ lớn tương đương, theo từng seed): với mỗi liều α, tỉ số ΔE76(rot_α)/ΔE76(dir_α) phải nằm trong [0,9; 1,1] ở cả 5 seed. Nếu không, liều α bị loại khỏi phân tích cặp và được báo cáo là không đạt.

Cổng cân bằng: với nhánh dir_α và rot_α, SMD trung bình 5 seed của độ bão hòa RGB và của L phải có |·| ≤ 0,2. Nếu không đạt, liều đó không được diễn giải. Cổng này được kiểm trước bằng thống kê ảnh (pilot, mục 9), không cần mô hình.

Tín hiệu gốc: CI 95% của AUROC gốc (liều 0) phải không chứa 0,5 để bộ đó được diễn giải. Bộ không đạt được báo cáo là "không có tín hiệu gốc".

Phân tích chính (đặt trước): α = 1,0, điểm năng lượng, cho từng bộ có màu (retina, PathMNIST, BloodMNIST). Hiệu D = AUROC(dir_1) − AUROC(rot_1), CI 95% t theo 5 seed. Biên tương đương δ = 0,05.

Quy tắc, theo thứ tự ưu tiên (áp dụng cho từng bộ):
1. Nếu không đạt cổng ΔE, cổng cân bằng, hoặc tín hiệu gốc: "không kết luận" (lý do ghi rõ).
2. Nếu CI(D) ⊂ [−δ, δ]: "không đặc hiệu theo hướng" (hướng về far không làm khác placebo, tức không tìm thấy vai trò riêng của hướng).
3. Nếu CI(D) ⊂ (−∞, −δ): "đặc hiệu theo hướng về far" (khớp hướng far giảm phát hiện nhiều hơn placebo vuông góc).
4. Nếu CI(D) ⊂ (δ, ∞): "đặc hiệu ngược" (placebo giảm nhiều hơn).
5. Các trường hợp còn lại: "không kết luận: CI quá rộng".

Các liều 0,25 và 0,5 là phân tích phụ, báo cáo đường liều (D theo α) nhưng không dùng để kết luận.

Hiệu chỉnh nhiều so sánh: phân tích chính gồm 3 bộ có màu; áp dụng hiệu chỉnh Holm trên 3 p-value. Bộ xám không đưa vào kết luận về hướng.

Shift: báo cáo cho mọi nhánh. Không dùng làm cổng. Hai shift khác nhau có ý nghĩa giữa dir và rot ở cùng liều được báo cáo như bằng chứng bổ sung.

## 5. Sức mạnh và giới hạn

- n = 5 seed. Biên δ = 0,05 được chọn để CI có thể nằm trọn trong [−δ, δ]. Với độ lệch chuẩn của D trên seed khoảng 0,02–0,03 (ước lượng từ kết quả chroma v3), nửa khoảng tin cậy 5 seed khoảng 0,03; có thể kết luận "không đặc hiệu" nếu hiệu thật nhỏ hơn 0,02.
- Nếu cổng cân bằng độ bão hòa không đạt ở α = 1 cho một bộ, kết luận về bộ đó là "không kết luận". Đây là hạn chế của lưới và được báo cáo như vậy.
- Mẫu far nhỏ (156–800 ảnh) và mẫu ID test 600 ảnh tạo sai số lấy mẫu trong AUROC; có thể làm tăng biến thiên giữa các seed.
- Phép khớp chỉ chỉnh trung bình và độ lệch chuẩn từng kênh, không khớp hình dạng phân phối. Cổng ΔE và cổng cân bằng bắt được một phần lệch này, không phải tất cả.
- Chỉ kiểm trên v3 (EfficientNet-B0) và năm bộ far. Không khái quát ra v4 hay các bộ khác.

## 6. Thay đổi so với đợt chroma v3 (khai báo để minh bạch)

- Placebo: từ xoay toàn bộ đám mây chroma của far theo θ ngẫu nhiên (theo seed) sang phép xoay 90° của độ dời trung bình, giữ σ đích. Lý do: bản cũ có độ lớn và σ không khớp, và hướng thay đổi theo seed.
- Cổng shift 0,5 ± 0,05 của v3 đã bị bỏ khỏi quy tắc quyết định. Lý do: cổng này loại PathMNIST vì shift 0,68, nhưng shift chính là một kết quả cần quan sát, không phải điều kiện của câu hỏi. Shift vẫn được báo cáo.
- Cổng ΔE mới: không có trong v3.
- Liều: thêm α = 0,25 và 0,5 (v3 chỉ có α = 1).
- Quy tắc quyết định: có thứ tự ưu tiên và biên tương đương, thay cho các quy tắc chồng chéo của v3.
- Kết quả v3 chroma và chroma_rot (đã có trong LOG) không thể so sánh trực tiếp với đợt này, vì placebo đã đổi.

## 7. Chi phí ước tính

Mỗi seed: khoảng 18 000 lượt suy luận CPU (EfficientNet-B0, 224 px) cho 3 bộ có màu (dir và rot ở 3 liều, cộng gốc) và 2 bộ xám (dir ở 3 liều). Khoảng 25 phút mỗi seed, khoảng 2–2,5 giờ cho 5 seed. Không có huấn luyện lại.

## 8. Đầu ra

- `dose_seed{s}.json`: mọi nhánh, liều, số liệu, ΔE, SMD, shift.
- `dose_summary.csv`: trung bình, CI, SMD, cổng ΔE, kết luận theo mục 4.
- `dose_response.csv`: D_α theo cặp, độ dốc theo α, p hiệu chỉnh Holm.

## 9. Kiểm khả thi trước khi chấm mô hình (pilot, chỉ thống kê ảnh)

Trước khi chấm mô hình, chạy pilot chỉ dùng ảnh: với mỗi bộ có màu (retina, PathMNIST), mỗi seed và mỗi liều, tính θ theo mục 2, dựng ảnh dir và rot, rồi báo cáo SMD độ bão hòa và L, tỉ số ΔE76 rot/dir, cos giữa hai độ dời, và chênh lệch độ lớn chroma. Pilot không dùng AUROC hay kết quả mô hình. Nếu pilot cho thấy cổng cân bằng không thể đạt ở α = 1, thiết kế được báo là không khả thi và không chấm mô hình.

## 10. Kết quả pilot và chuyển sang đối chứng khử màu hai phía

Pilot (mục 9, chỉ thống kê ảnh, 5 seed, C:\data\bcn20000\dose_pilot.csv):
- Retina: cổng ΔE và độ lớn đạt. Placebo đạt cân bằng độ bão hòa ở α = 1 (−0,12), nhưng nhánh dir không cân bằng: độ bão hòa 0,33 và L lệch +1,16 ở mọi liều (nhánh chroma giữ L, bộ far sáng hơn ID).
- PathMNIST: không khả thi. Chênh lệch độ lớn chroma nhỏ nhất với |cos| ≤ 0,3 là 16–21%, vượt 2%.
- BloodMNIST: tỉ số ΔE 0,84–0,88, ngoài [0,9; 1,1]; độ bão hòa và L lệch.
Kết luận theo mục 9: thiết kế placebo theo màu không khả thi, không chấm mô hình cho thiết kế này.

Thay thế: đối chứng khử màu hai phía. Đưa cả ID và far về thang xám (luminance, ba kênh giống nhau), rồi đo AUROC.
- auc_gg = AUROC(ID xám so với far xám). Không có placebo nên không có vấn đề cân bằng sắc độ.
- auc_gg_lum = như trên nhưng ID xám được khớp histogram độ sáng với far xám. Kiểm tra độ sáng có phải nguyên nhân không.
- Đối chiếu với auc_orig (ID màu so với far màu).

Quy tắc quyết định (đặt trước):
1. Nếu CI của auc_orig chứa 0,5: không có tín hiệu để giải thích.
2. Phụ thuộc màu: CI(auc_gg) ⊂ (0,45; 0,55). Tín hiệu mất khi bỏ màu.
3. Không phụ thuộc màu: CI(auc_gg) có cận dưới > 0,55. Tín hiệu còn khi bỏ màu.
4. Phụ thuộc độ sáng: CI(auc_gg_lum) ⊂ (0,45; 0,55) trong khi auc_gg không ở trường hợp 3.
5. Trường hợp khác: không kết luận.
Ghi chú cân bằng: SMD độ sáng giữa ID xám và far xám được báo cáo; không dùng làm cổng vì đây là thuộc tính của bộ far.
Bộ xám (breastmnist, pneumoniamnist) có thể đưa vào như kiểm tra nội bộ, vì far đã là xám.

## 11. Quy tắc đối chứng khử màu hai phía, v0.4 (thay thế quy tắc mục 10)

Lý do thay thế: đánh giá độc lập (workflow ma21-review-gray-both-control) cho thấy quy tắc v0.3 sai:
- Ngưỡng tuyệt đối CI(gg) > 0,55 cho nhãn "không phụ thuộc màu" ngay cả khi bộ mất ~76% tín hiệu (võng mạc: 0,42 → 0,10 trên phần vượt 0,5).
- Không đo shift: bỏ màu ID tự đổi điểm (ID xám là OOD với mô hình), nên gg thấp có thể không phải do màu.
- gg_lum không có vai trò miễn trừ nhãn màu, và nhãn áp cho bộ xám dù không có phép biến đổi far.
Quy tắc v0.3 không được dùng để kết luận. Quy tắc v0.4 được định nghĩa trước khi chạy lại.

Đại lượng (năng lượng sau T, theo từng seed):
- A_gốc = AUROC(ID màu, far màu). A_gg = AUROC(ID xám, far xám). A_gl = AUROC(ID xám khớp độ sáng, far xám).
- shift_id = AUROC(ID màu, ID xám): mức đổi điểm do chính phép khử màu lên ID (0,5 là không đổi).
- Tỉ lệ giữ tín hiệu R = (A − 0,5)/(A_gốc − 0,5), tính theo từng seed rồi lấy CI t 95% (n = 5).

Quy tắc, theo thứ tự ưu tiên:
1. Bộ xám (breastmnist, pneumoniamnist): không áp dụng.
2. CI(A_gốc) chứa 0,5: không có tín hiệu gốc.
3. CI(shift_id) không nằm trọn trong (0,45; 0,55): không kết luận, bỏ màu tự đổi điểm ID.
4. CI(R_gg) có cận dưới ≥ 0,8: không phụ thuộc màu (giữ ≥ 80% tín hiệu).
5. CI(R_gg) có cận trên ≤ 0,2: phụ thuộc màu (mất ≥ 80%). Nếu CI(R_gl) có cận dưới ≥ 0,8, ghi thêm: phụ thuộc độ sáng.
6. Trường hợp khác: không kết luận, giữ một phần tín hiệu.

Ghi chú: shift_far đo đổi điểm của far khi khử màu (0,5 với bộ xám). Được báo cáo, không dùng làm cổng.

## 12. Mô hình huấn luyện trên ảnh xám (v5g), v0.5: giải quyết ảnh hưởng của màu

Trạng thái: quy tắc đặt trước v0.5 (thay thế v0.4 của mục 12 bản trước, chưa chạy lần nào), sau đánh giá độc lập (workflow ma21-review-gray-model-design, 8 phát hiện đã xác minh).

Lý do: mục 10–11 không tách được "phát hiện cần màu" khỏi "bỏ màu làm đổi điểm" vì mô hình huấn luyện trên ảnh màu coi ảnh xám là OOD. Nếu mô hình được huấn luyện trên ảnh xám, ảnh ID xám là miền gốc của nó, nên không còn dịch chuyển do bỏ màu.

### Thiết lập
- EfficientNet-B0, trọng số khởi tạo ImageNet (hạn chế: học từ ảnh màu).
- Chia tổn thương theo seed như v3; HOLDOUT = AK; EPOCHS = 30; PATIENCE = 5; các siêu tham số giữ như v3.
- GRAYSCALE = 1: ảnh train, test ID và far đều đổi sang luminance, nhân ba kênh. Đối chứng khớp (matched) bị tắt vì độ bão hòa bằng 0 với mọi ảnh xám. GRAY_CONTROL và FAR_GRAY trở thành so sánh tự thân: cột gốc và cột xám giống hệt nhau nên AUROC bằng 0,5 do điểm hòa (đã kiểm chứng trên 91.808 cặp). Không dùng trong quyết định. Cần tắt khi GRAYSCALE = 1 (chưa sửa code).
- 10 seed (0–9). Lý do: với SD seed 0,043 (võng mạc v3), nửa khoảng CI 10 seed là 0,031, đủ để CI có thể nằm trong (0,45; 0,55) khi không có tín hiệu.
- Bộ far: retinamnist, pneumoniamnist, breastmnist (toàn bộ test); bloodmnist và pathmnist (800 ảnh test, seed 0, tạo bởi code/make_farsub.py). MD5 tệp đã tải lên: retina 84dea4cffe5416fb52f000b7e54c0425; pneumonia 550e68abc821d4f96037458b7fb6656c; breast 2c432592eb2187694718b4f843443807; blood 78c993140cd733dced71d39e1b378074; path 634f6322296b1a8070cee5c6956ec03b.

### Đại lượng
- A_xám: AUROC năng lượng sau T (ID test xám so với far xám), trung bình và CI t 95% trên 10 seed.
- A_màu: AUROC năng lượng của mô hình màu v3 trên cùng bộ far, trung bình 5 seed (tham chiếu cố định): retina 0,917 [0,863; 0,971], pathmnist 0,856 [0,731; 0,981], bloodmnist 0,653 [0,500; 0,806].
- Tỉ lệ giữ tín hiệu: R = (A_xám − 0,5)/(A_màu − 0,5). Các cận của CI_xám được chuyển thành cận của R theo cùng công thức. Đây là so sánh giữa hai mô hình (không ghép cặp), nên là đại lượng gần đúng; báo cáo kèm điều này.
- BA_xám: độ chính xác cân bằng trung bình trên ID test (10 seed). BA_màu = 0,562 (v3, 5 seed).
- A_AK,xám: AUROC AK (OOD lớp giữ lại) năng lượng trong miền xám. A_AK,màu = 0,605 (v3).

### Quy tắc quyết định (đặt trước; thứ tự ưu tiên)
1. Cổng hợp lệ của mô hình: BA_xám ≥ BA_màu − 0,05 (≥ 0,512) VÀ A_AK,xám (trung bình) ≥ 0,55. Không đạt: "không kết luận: mô hình xám không đạt đối chứng".
2. Cổng tín hiệu tham chiếu: CI của A_màu không chứa 0,5 (cận dưới > 0,5). Không đạt (ví dụ bloodmnist, cận dưới = 0,500): "không có tín hiệu tham chiếu để so sánh".
3. Nếu cận dưới của R ≥ 0,8: "không phụ thuộc màu: giữ ≥ 80% tín hiệu khi không có màu".
4. Nếu cận trên của R ≤ 0,2: "phụ thuộc màu: mất ≥ 80% tín hiệu khi không có màu".
5. Các trường hợp khác: "không kết luận: giữ một phần tín hiệu hoặc CI rộng".

Chỉ áp dụng cho ba bộ có màu (retina, pathmnist, bloodmnist). Không hiệu chỉnh nhiều so sánh vì không có kết luận p-value; mỗi bộ có một kết luận độc lập, báo cáo đầy đủ.

### Điều cần nêu khi báo cáo (bất đối xứng)
- Kết luận "phụ thuộc màu" mạnh hơn kết luận "không phụ thuộc màu". Lý do: độ phân giải không cân bằng (far 224 px được phóng lên 256 px; ID thu nhỏ từ ảnh gốc lớn). Nếu tín hiệu còn lại, có thể là do chênh lệch độ phân giải chứ không phải cấu trúc ngữ nghĩa. Nếu tín hiệu mất, vẫn có chênh lệch độ phân giải nên đó không phải lý do; tín hiệu mất là do bỏ màu.
- Độ sáng: SMD độ sáng trung bình giữa ID xám và far xám được tính và lưu (far_lum_smd). Đây là thông tin, không là cổng. Độ sáng và tương phản là một phần của tín hiệu có thể còn lại.
- Chỉ kết luận được về vai trò của màu. Không loại trừ các thống kê mức thấp khác (kết cấu, tương phản).
- Trọng số khởi tạo ImageNet học từ ảnh màu có thể mang đặc trưng màu.
- Tham chiếu A_màu dùng mẫu far khác (v3 dùng toàn bộ bloodmnist và pathmnist; v5g dùng 800 ảnh). Chênh lệch này là nhỏ với AUROC nhưng cần ghi nhận.

### Chi phí
Kaggle GPU T4 x2: khoảng 20 phút chuẩn bị dữ liệu và khoảng 35 phút huấn luyện và đánh giá 10 seed.


## 13. Hướng B: giảm learning rate, đăng ký trước khi chạy (v5h và v3h, 2026-10-19)

**Lý do.** Đánh giá độc lập (LOG, 2026-10-19) xếp B là có điều kiện. Đợt này kiểm tra xem BA thấp của mô hình xám có phải do lịch huấn luyện không.

**Thay đổi duy nhất so với v5g: LR = 1e-4 (v5g và v3 dùng 3e-4).**
- Giá trị được chọn trước, không chọn theo kết quả, theo bộ far, hay theo AK.
- Chỉ có một giá trị thay thế. Hạn chế: không quét LR, nên không loại trừ được một LR khác tốt hơn.

**Hai nhánh, cùng LR, cùng tập far con (farsub):**
- v5h, mô hình xám: GRAYSCALE = 1, EfficientNet-B0, 10 seed (0–9), LR = 1e-4.
- v3h, mô hình màu đối chứng: GRAYSCALE = 0, EfficientNet-B0, 5 seed (0–4), LR = 1e-4. Lý do: BA tham chiếu phải cùng LR để so sánh công bằng.

**Tham chiếu màu thay đổi (khai báo trước):** BA_màu và A_màu lấy từ v3h, không dùng v3 (3e-4) và dose_v3. Lý do: cùng LR, cùng tập far. Điều này thay đổi tham chiếu, không thay đổi quy tắc.

**Quy tắc quyết định giữ nguyên mục 12:**
1. Cổng mô hình: BA_xám ≥ BA_màu − 0,05 và AK_xám ≥ 0,55. Không đạt thì "không kết luận".
2. Cổng tín hiệu tham chiếu: CI của A_màu có cận dưới > 0,5.
3. Quyết định theo R = (A_xám − 0,5)/(A_màu − 0,5), với các cận của CI.

**Đã sửa trước khi chạy:** đối chứng xám bị tắt khi GRAYSCALE = 1 (mục 12 đã ghi lại lỗi). Không ảnh hưởng quy tắc quyết định.

**Diễn giải.** Nếu v5h vẫn không đạt cổng, kết quả là "không kết luận", và nhánh huấn luyện không phải nguyên nhân chính. Nếu đạt cổng, mới so R. Không đổi cổng hay quy tắc sau khi thấy kết quả.

**Chi phí.** Khoảng 45 phút đồng hồ trên 2 GPU (xám 10 seed ~27 phút, màu 5 seed ~14 phút), chạy thành hai phiên Kaggle.


## 14. A3, phiên bản 2 (sửa theo phản biện trước khi chạy, 2026-10-20): can thiệp bất biến độ sáng và tương phản

**Câu hỏi.** Khi mô hình màu không thể dựa vào độ sáng tuyệt đối và tương phản, phát hiện far-OOD còn giữ bao nhiêu tín hiệu so với đối chứng cùng cấu hình?

**Lý do.** Chẩn đoán đã đúng khung (LOG, 2026-10-20): độ sáng tách võng mạc (AUROC 0,918), breast (0,930), BloodMNIST (0,824 theo chiều đảo) và PathMNIST (0,636 theo chiều đảo). Chẩn đoán đơn biến không cho biết mô hình có dùng độ sáng hay không. A3 kiểm bằng can thiệp lên huấn luyện, và kiểm không huấn luyện trên đầu ra đã có.

**Mô hình.**
- J (can thiệp): giống v3 về mọi cấu hình (EfficientNet-B0, LR 3e-4, EPOCHS 30, PATIENCE 5, HOLDOUT AK, WD 1e-4), thêm tăng cường trên ảnh train. Tăng cường áp dụng lên khung 256 px trước khi cắt: độ sáng nhân b ~ U(0,6; 1,4) rồi kẹp [0, 1]; tương phản c ~ U(0,7; 1,3) quanh trung bình luminance của chính ảnh, kẹp [0, 1]. Val, ID test, AK và far không đổi.
- B (đối chứng): cùng cấu hình, không tăng cường, chạy lại trên mã này. Seed 0 đến 9.
- Khung đánh giá: far được phóng 224 lên 256 bilinear rồi cắt giữa 224, như mô hình nhìn thấy. ID là X_train_256 cắt giữa 224.

**Bộ và đại lượng.**
- Bộ chính: võng mạc và PathMNIST. BloodMNIST, breast và pneumonia chỉ báo cáo, không quyết định.
- A_J, A_B: AUROC năng lượng (energy_ts), trung bình 10 seed, CI t 97,5% (Bonferroni cho 2 bộ chính). Hai nhóm là độc lập, không ghép cặp.
- R = (A_J − 0,5) / (A_B − 0,5). CI của R lấy từ các cận của A_J và A_B, như mục 12.
- Làm rõ, ghi sau khi có kết quả A3 (2026-10-21, sau rà soát độc lập): cách đọc đã dùng để tính kết quả A3 là mẫu số là điểm ước lượng A_B, chỉ cận của A_J đổi, đúng mục 12. Cách đọc còn lại (cả hai cận cùng đổi) cho cận dưới R của PathMNIST là 0,742 thay vì 1,002, nên quyết định PathMNIST đổi thành "không kết luận". Cả hai cách được ghi trong analysis/jit_a3_rev2.
- Quyết định (2026-10-21, người nghiên cứu chốt): Cách A (mục 12) là phân tích chính, đúng như đã đăng ký. Cách B (cả hai cận cùng đổi) là phân tích độ nhạy, báo cáo cùng kết quả chính. Lý do: Cách B là cặp cực trị ngược chiều đồng thời, không phải CI thống kê của tỉ số, nên bảo thủ quá mức; đổi cách sau khi thấy số là hậu nghiệm và không được phép. Vì vậy kết luận PathMNIST "còn tín hiệu" chỉ đúng theo Cách A, và phải báo cáo kèm Cách B: cận dưới 0,742, không đạt ngưỡng 0,8.
- D1 (độ nhạy độ sáng của mỗi mô hình): trên ID test, E là energy_ts, D1 = trung bình theo seed của [mean|E(ID × 0,8) − E(ID)| + mean|E(ID × 1,2) − E(ID)|] / 2, chia cho SD của E(ID) của chính mô hình đó. Chuẩn hóa theo SD để không lẫn với thay đổi thang logit.

**Cổng (đánh giá theo thứ tự; không đạt thì không kết luận).**
- C0 (có tín hiệu gốc): CI_cận dưới(A_B) > 0,60 cho bộ đó. Ngưỡng 0,60 là lựa chọn, báo cáo kèm R không cổng để người đọc tự đánh giá.
- G1 (ID giữ được): hiệu BA_J − BA_B, trung bình 10 seed, ≥ −0,03; và không seed nào giảm quá 0,05 (ghép cặp theo seed).
- G2 (AK giữ được): trung bình AK_J ≥ trung bình AK_B − 0,05.
- G3 (can thiệp có tác dụng): S = D1_J / D1_B ≤ 0,5.

**Quy tắc quyết định (chỉ khi C0, G1, G2, G3 đạt, cho từng bộ chính).**
- "Còn tín hiệu sau bất biến": cận dưới CI(R) ≥ 0,8.
- "Giảm mạnh khi bất biến": cận trên CI(R) ≤ 0,2.
- Còn lại: không kết luận.

**Kiểm không huấn luyện (CPU, trên đầu ra đã có; báo cáo, không đổi quyết định).**
- K1: chia ảnh ID test và far của mỗi bộ (đưa về khung mô hình) thành 5 bin theo luminance trung bình, bin theo phân vị của gộp. Tính AUROC năng lượng của B và J trong từng bin, trung bình có trọng số theo số ảnh; ghi số ảnh mỗi bin. Nếu AUROC trong bin vẫn > 0,6, tín hiệu không chỉ là chênh lệch độ sáng toàn cục. Cột gt_0_6 của k1_summary ghi điều này cho AUROC có trọng số. Bin có ít hơn 20 ảnh mỗi nhóm bị bỏ; số bin dùng ghi ở bins_used.

**Giới hạn ghi trước.**
1. Ngoại suy: võng mạc có độ sáng trung bình khoảng 0,61 của ID (khung mô hình: 85 so với 140). Khoảng tăng cường bao đến 0,6, nên đuôi tối hơn không được che. "Còn tín hiệu" chỉ nghĩa là trong khoảng đã thấy khi huấn luyện.
2. Tương phản được kiểm soát; bão hòa không can thiệp, không kiểm soát.
3. Độ phân giải không kiểm soát: far 224 px được phóng lên 256, mờ hơn ID.
4. Không có đối chứng placebo. "Giảm" không chứng minh tính đặc hiệu với độ sáng; kết quả chỉ mô tả mức giữ tín hiệu khi bất biến độ sáng và tương phản.
5. Một kiến trúc, một ID, hai bộ far chính; không quét cường độ.
6. CI phản ánh biến thiên giữa seed huấn luyện; hai nhóm không ghép cặp hoàn toàn.
7. Hiệu lực công suất: MDE sẽ được báo cáo cùng kết quả; nếu CI quá rộng, kết luận là không kết luận.

**Diễn giải được phép.**
- Dương: "AUROC_J giữ R% (CI) khi mô hình bất biến độ sáng ±40% và tương phản ±30%, khung 224 px."
- Âm: "AUROC giảm còn R% khi thêm bất biến độ sáng và tương phản." Không viết "đặc hiệu với độ sáng" hay "do ngữ nghĩa".
- Không kết luận: báo cáo kèm MDE.

**Chi phí.** Hai phiên Kaggle (J và B), chạy song song nếu hàng đợi cho phép; mỗi phiên khoảng 35 đến 40 phút với 10 seed trên 2 GPU, cộng hàng đợi. K1 chạy trên CPU.

**Không làm.** Không đổi cổng, quy tắc, cường độ hay bộ far sau khi thấy kết quả. Không chọn b, c hay ngưỡng theo bộ far hoặc AK.

**Ghi chú lịch sử.** Phiên bản 1 (cùng ngày) dùng ngưỡng tuyệt đối AUROC_J ≥ 0,60 và G3 không chuẩn hóa. Hai phản biện (một về phương pháp, một về khả thi) cho kết luận REVISE, nên phiên bản 1 không được chạy.


## 15. Claim bài báo và kế hoạch đăng ký (chọn claim kiểu đánh giá, 2026-10-20; chưa chạy)

**Lựa chọn.** Bài báo là một nghiên cứu đánh giá, không phải đề xuất phương pháp OOD mới. Lý do: mọi kết quả hiện có cho thấy tín hiệu far-OOD của mô hình da liễu liên quan đến thống kê mức thấp, nhưng chưa đủ để nói phương pháp nào tốt hơn. Kết quả âm có đăng ký trước vẫn có giá trị khoa học.

**Claim chính (C1).** Với mô hình phân loại da liễu, điểm OOD (năng lượng, sau temperature scaling) và tín hiệu far-OOD phụ thuộc đáng kể vào thống kê ảnh mức thấp (độ sáng, tương phản, độ phân giải). Mức phụ thuộc khác nhau theo bộ far và tập ID.

**Claim phụ.**
- C2: khung không đồng nhất giữa far và ID làm lệch AUROC độ sáng khoảng 0,05 (0,970 khung 224 gốc so với 0,918 khung mô hình). Đề xuất giao thức đồng nhất khung.
- C3: sau can thiệp bất biến độ sáng và tương phản (mục 14), tín hiệu của võng mạc và PathMNIST giữ phần lớn mức ban đầu. Báo cáo cùng với kiểm placebo và kiểm độ phân giải.

**Không claim.** Không claim phương pháp OOD tốt hơn. Không claim tín hiệu là "ngữ nghĩa" nếu chưa có placebo đạt. Không claim tổng quát từ một tập ID hay một kiến trúc.

**Thí nghiệm dự kiến (chưa chạy, cần duyệt GPU).**
- E1, kiểm độ phân giải: ID đi qua chuỗi 224 → 256 → cắt giống far khi đánh giá; và/hoặc augmentation làm mờ ngẫu nhiên khi huấn luyện. So sánh AUROC năng lượng với mô hình B. Quyết định: nếu AUROC thay đổi quá 0,05 thì độ phân giải là nguồn nhiễu đáng kể.
- E2, placebo hình học: augmentation xoay, co và cắt với cường độ tương đương (cùng mức giảm BA như J). Nếu J làm giảm AUROC nhiều hơn placebo ở cùng mức BA thì ủng hộ tính đặc hiệu với độ sáng và tương phản. Nếu không, kết luận chỉ là giảm chung.
- E3, ID thứ hai: PAD-UFES-20 (CC BY 4.0 theo trang Kaggle; trang Mendeley gốc chưa xác nhận được). Ánh xạ 6 lớp sang nhãn chung: ACK→AK, BCC→BCC, MEL→MEL, NEV→NV, SCC→SCC, SEK→BKL, BOD gộp vào SCC. Mô hình B và J trên PAD với cùng quy trình. Lưu ý: PAD có ảnh điện thoại, khác miền với BCN20000.
- E4, kiến trúc: ConvNeXt-T với cùng cặp J và B.
- E5, công suất: tăng số seed võng mạc từ 10 lên 20, cho cặp J và B.

**Quy tắc quyết định (sẽ được ghi chi tiết trước khi chạy từng thí nghiệm).** Dùng R và CI như mục 14, với cổng tín hiệu gốc, cổng BA và AK. Không đổi quy tắc sau khi thấy kết quả. Báo cáo cả kết quả âm.

**Rủi ro và giới hạn đã biết.** Chỉ hai tập ID; PAD có ánh xạ nhãn không hoàn hảo; ảnh điện thoại có thể có thống kê khác hẳn; chưa có kết quả E1 đến E5; giới hạn về độ phân giải và placebo như mục 14.

**Ngày và phạm vi.** Mục này thay cho mục 14 về mục tiêu bài báo. Mục 14 vẫn là đăng ký thí nghiệm A3 đã chạy.


## 16. Đăng ký E1, E2, E3, E5 (2026-10-21, trước khi chạy)

Các thí nghiệm này thực hiện kế hoạch mục 15. Dùng cùng khung đánh giá, cùng cổng và cùng quy tắc R của mục 14 trừ khi ghi khác. Mọi thay đổi quy tắc sau khi thấy kết quả đều không hợp lệ.

**Chung.**
- Bộ far chính: võng mạc và PathMNIST (farsub, khung mô hình). Breast và pneumonia chỉ báo cáo.
- A (AUROC năng lượng), CI t hai phía 97,5% cho mọi bộ far, cùng quy ước mục 14 (ppf 0,9875, đã hiệu chỉnh cho hai bộ chính). R = (A − 0,5) / (A_đối chứng − 0,5). Mẫu số là điểm ước lượng của đối chứng; các cận của tử số lấy theo cùng công thức, như mục 12. Cách đọc khác (cả hai cận cùng đổi) chỉ báo cáo làm độ nhạy.
- Chốt áp dụng chung (2026-10-21, người nghiên cứu duyệt, không đổi lại): Cách A là phân tích chính cho A3 và E1 đến E5. Cách B được báo cáo song song là độ nhạy, trong cùng decision.csv (cột R_lo_both_bounds và R_hi_both_bounds). Không gọi Cách B là Bonferroni và không gán cho nó ý nghĩa xác suất mức α.
- Cổng C0 (tín hiệu gốc), G1 (BA), G2 (AK) và G3 (độ nhạy độ sáng) như mục 14, với đối chứng tương ứng của từng thí nghiệm.
- Mọi kết quả được báo cáo kể cả khi âm. Không chạy thêm để tìm kết quả thuận lợi.
- Làm rõ, ghi trước khi có kết quả: ở E1, cổng C0 là tín hiệu của B (đối chứng, như định nghĩa mục 14), G1 và G2 của B_R so với B, G3 không áp dụng vì không có cặp can thiệp độ sáng; R của B_R được báo cáo. Ở E2, cổng là điều kiện hợp lệ của P và C0 của B; G1 đến G3 không dùng vì P không đổi độ sáng. Ở E3, không có K1, vì K1 cần mảng ảnh PAD chưa dựng. Ở E5, quy tắc mục 14 trên 20 seed, kèm kết quả 10 seed để so độ rộng CI.
- Sửa trước khi có kết quả (2026-10-21, sau rà soát độc lập): (i) E2 thêm điều kiện khớp BA với J, theo mục 15; (ii) E1 áp RES_MATCH cho cả holdout AK (xem mục E1); (iii) ghi rõ cách đọc R của mục 12 ở trên.

**E1, kiểm soát độ phân giải.**
- Mô hình B_R: giống B (mục 14) nhưng mọi ảnh BCN (ID train, val, test và holdout AK, cùng nguồn) đi qua chuỗi 224 → 256 bilinear trước khi chuẩn hóa. Far không đổi. Đây là biến đổi duy nhất. Sửa trước khi có kết quả: bản đầu chỉ ghi ID. Mã áp cho cả AK, vì AK cùng nguồn BCN; nếu không áp cho AK thì AK sắc hơn ID trong B_R và tạo một chênh lệch độ phân giải mới.
- So sánh A(B_R) với A(B) cho từng bộ far chính, ghép theo seed 0 đến 9.
- Quyết định: nếu |A(B_R) − A(B)| ≥ 0,05 và CI của hiệu không chứa 0 ở bộ chính thì độ phân giải là nguồn nhiễu đáng kể. Khi đó các kết luận của mục 14 phải kèm cảnh báo này. Nếu không, độ phân giải không đổi kết luận ở mức này.
- Giới hạn: chỉ kiểm chuỗi 224 → 256 với một kiểu nội suy. Không loại trừ các khác biệt tần số khác.

**E2, placebo hình học.**
- Mô hình P: giống B nhưng khi huấn luyện, mỗi ảnh xoay ngẫu nhiên θ ~ U(−30°, 30°) với phản chiếu biên, không đổi độ sáng và tương phản. Đây là đối chứng để kiểm tính đặc hiệu.
- Điều kiện hợp lệ: BA(P) − BA(B) ≥ −0,03 và |BA(P) − BA(J)| ≤ 0,03 (trung bình 10 seed). Điều kiện thứ hai khớp mức giảm BA của J, như mục 15 yêu cầu; bản đầu của mục 16 thiếu điều kiện này. Nếu không đạt một trong hai, E2 không kết luận được về tính đặc hiệu. Giới hạn: placebo chỉ xoay, không co và cắt như mục 15 mô tả.
- So sánh giảm AUROC của J và của P so với B: d_J = A(B) − A(J) và d_P = A(B) − A(P), ghép theo seed.
- Quyết định: nếu d_J − d_P ≥ 0,05 với CI không chứa 0 thì tác động của J đặc hiệu hơn tác động hình học, ủng hộ tính đặc hiệu với độ sáng và tương phản. Nếu |d_J − d_P| < 0,05 thì tác động không đặc hiệu. Trường hợp còn lại (d_J − d_P ≥ 0,05 nhưng CI chứa 0, hoặc d_J − d_P ≤ −0,05) không thuộc hai nhánh trên: báo cáo nguyên văn, không gán nhãn.
- Giới hạn: xoay làm mờ nhẹ do nội suy. Báo cáo độ sắc nét (trung bình gradient) của ảnh P so với ảnh gốc như một đại lượng kèm.

**E3, tập ID thứ hai (PAD-UFES-20).**
- Dữ liệu: metadata và ảnh PNG từ bản gương Kaggle mahdavi1202/skin-cancer (license ghi CC BY 4.0, chưa xác nhận từ Mendeley). Khóa tổn thương (patient_id, lesion_id). Ánh xạ nhãn theo mục 15. Lớp giữ lại: ACK (AK).
- Thiết kế: J_pad (AUG_BRIGHT 0,4, AUG_CONTRAST 0,3) và B_pad (không tăng cường), cùng cấu hình như mục 14, 10 seed, cùng bộ far.
- Cổng và quy tắc: như mục 14 với đối chứng B_pad. Bộ chính vẫn là võng mạc và PathMNIST, cùng với đối chứng BA và AK của PAD.
- Lưu ý: MEL có 52 ảnh, nên chỉ báo cáo theo lớp, không kết luận. Đây là kiểm tính tổng quát theo tập ID, không phải kiểm lại cùng dữ liệu.
- Điều kiện trước khi công bố: xác nhận license PAD-UFES-20 từ nguồn gốc.

**E5, công suất võng mạc.**
- Thêm 10 seed (10 đến 19) cho J và B trên BCN20000, cùng cấu hình mục 14. Kết hợp với 10 seed đã có để thành 20 seed mỗi nhánh.
- Quyết định: áp dụng quy tắc mục 14 trên 20 seed. Báo cáo cả kết quả 10 seed và 20 seed để thấy CI thu hẹp hay không.

**E4 (ConvNeXt-T).** Đăng ký chi tiết ở mục 17, ghi trước khi chạy.

## 17. Đăng ký E4, kiến trúc ConvNeXt-T (2026-10-27, trước khi chạy)

E4 thực hiện mục 15 (kiến trúc ConvNeXt-T với cùng cặp J và B). Đăng ký này được ghi sau khi E1, E2, E3 và E5 đã có kết quả nhưng trước khi có bất kỳ kết quả E4 nào. Mục 16 áp dụng cho E4 trừ khi ghi khác ở đây.

- **Mô hình:** ConvNeXt-Tiny, khởi tạo ImageNet (IMAGENET1K_V1). Đầu phân loại: LayerNorm, dropout 0,3, tuyến tính 768 → số lớp. Đúng như code/ood_calibration.py khi BACKBONE = convnext_t.
- **Cặp J và B:** giống E5. J có AUG_BRIGHT 0,4 và AUG_CONTRAST 0,3. B không tăng cường (0 và 0). Cả hai bật BRIGHT_SENS để đo D1.
- **Cấu hình khác:** giữ như mục 14: BCN20000, holdout AK, EPOCHS 30, PATIENCE 5, GRAYSCALE 0, RES_MATCH 0, AUG_ROT 0, DATASET bcn. LR 3e-4 (mặc định của code, không chỉnh riêng cho ConvNeXt). WD 1e-4, batch 64 mỗi GPU.
- **Seed:** 0 đến 9 cho mỗi nhánh, tổng 20 lần huấn luyện. Không thêm seed sau khi thấy kết quả.
- **Quy tắc quyết định:** mục 14 và 16 áp dụng nguyên văn. Cổng C0 trên B. G1 (BA), G2 (AK), G3 (S = D1_J / D1_B). R Cách A là phân tích chính, Cách B là độ nhạy. Bộ chính: võng mạc và PathMNIST. Ba bộ còn lại chỉ báo cáo.
- **Phạm vi kết luận:** chỉ nói về ConvNeXt-Tiny với cấu hình này. Không claim tổng quát cho backbone khác. Seed lỗi (NaN hoặc phiên bị dừng) được ghi lại, không thay bằng seed khác.
- **Giới hạn:** LR không được tinh chỉnh cho ConvNeXt. Mã này chưa chạy ConvNeXt trên cấu hình hiện tại, nên seed đầu tiên có thể lộ lỗi; kiểm tra ConvNeXt cục bộ chỉ xác nhận cấu trúc mô hình.
