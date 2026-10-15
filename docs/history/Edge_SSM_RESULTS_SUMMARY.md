# Edge_SSM — tổng hợp kết quả trên dữ liệu WHAR

_Tự sinh bởi `build_summary.py` lúc 2026-10-12 15:04; chạy lại: `cd D:\Projects\Edge_SSM; uv run python build_summary.py`_

Thư mục `D:\Projects\Edge_SSM` (uv, Python 3.12, torch cu126, `uv.lock` chép từ `HAR/paper-era/env` → cùng phiên bản thư viện). Runner `run_whar.py` dùng nguyên harness WHAR Arena v1 thời điểm paper (cùng split, cùng vòng train, cùng metric) và cache `HAR/datasets_paper_era/`. Trang có biểu đồ: `report/index.html` (`build_report.py`) và `Edge-SSM atlas.html`; ghi chép chi tiết: `NEW_ARCHITECTURES.md`, `RESULTS.md`, `EDGE_SSM_SUMMARY.md` §6.

## 0. Cấu trúc mạng

Hình sinh bởi `build_arch_figures.py` (vết tín hiệu tính bằng chính các toán tử trong `edge_ssm/`).

**Khung chung.** Mọi họ mới dùng cùng một khung với S4D và Mamba: encoder tuyến tính → 3 khối → trung bình theo thời gian → LayerNorm → Linear ra 27 lớp. Trong mỗi khối chỉ ô MEMORY là khác nhau; u được đưa thẳng sang cạnh đầu ra bộ nhớ và có residual.

![Khung chung của các họ mới](docs/arch/skeleton.svg)

**QuantileNet — cấu trúc đầy đủ.** Encoder 6 → 32, ba QuantileBlock (mỗi khối 6,528 tham số: LayerNorm, trộn vào 32 → 32, 32 kênh × 4 bộ theo dõi phân vị với τ, η học được, ghép [u, q] thành 160, trộn ra 160 → 32, GELU, residual), trung bình theo thời gian, LayerNorm, Linear 32 → 27 — tổng 20,763. Một bộ theo dõi là bộ so sánh u_t < q? chọn bước +ητ hoặc −η(1−τ) rồi cộng vào thanh ghi: không có phép nhân khi chạy.

![QuantileNet](docs/arch/quantilenet.svg)

**HysteresisNet selective + clock — cấu trúc đầy đủ.** Thêm kênh đồng hồ (dốc −1 → +1 theo vị trí trong cửa sổ) vào 6 kênh cảm biến, encoder 7 → 32, ba HysteresisBlock (mỗi khối 7,456: như trên nhưng bộ nhớ là 4 play operator mỗi kênh, độ rộng r_t = softplus(ρ + s_t) với s_t từ nhánh Linear select), tổng 23,579. Một play operator là hai phép cộng (u ± r) và hai phép so sánh (max rồi min) với giá trị trước: không phụ thuộc tốc độ.

![HysteresisNet selective + clock](docs/arch/hystnet_sel_clock.svg)

**So với các họ kiến trúc khác** (cùng một ngữ pháp hình: phép cập nhật một bước và thanh ghi trạng thái z⁻¹):

![So sánh cách nhớ của các họ kiến trúc](docs/arch/families.svg)

| Họ | Trạng thái nhớ gì | Cập nhật mỗi bước | Phụ thuộc tốc độ | Nhân trong bộ nhớ | Huấn luyện song song |
|---|---|---|---|---|---|
| MLP / ANN | không có — cả cửa sổ là một vector | — | có (gắn vị trí mẫu) | — | có |
| KAN | không có — hàm học được trên mỗi cạnh | — | có | — | có |
| CNN | k mẫu gần nhất | tích chập với kernel học được | có | có | có |
| Transformer | toàn bộ (k, v) đã qua, lớn dần | attention softmax | có | nhiều | có |
| RNN (LSTM / GRU) | vector ẩn h | hàm phi tuyến có cổng của (h, x) | có | có | không (tuần tự) |
| CfC / LTC | vector h liên tục | ODE, τ phụ thuộc đầu vào | có | có | dạng đóng / giải ODE |
| SNN | điện thế màng v | tích phân rò, phát xung khi v > θ | có | ít | thường tuần tự |
| SSM (S4D / Mamba) | các mode suy giảm / dao động tuyến tính | h = A h + B x (Mamba: A, B, C theo x) | có | có | có (FFT / scan) |
| **HysteresisNet** | cực trị gần nhất vượt ± r | clamp(y, u − r, u + r) | **không** | **không** | **có** (scan trên monoid clamp) |
| **QuantileNet** | phân vị chạy của các giá trị gần đây | so sánh rồi cộng bước cố định | có (bước theo mẫu) | **không** | không (tuần tự) |

Điểm khác cốt lõi: hai họ mới giữ khung khối của SSM (trộn kênh ở ngoài, bộ nhớ theo từng kênh), nhưng thay phép đệ quy tuyến tính h = A h + B x bằng một phép đệ quy **phi tuyến không nhân**: clamp (HysteresisNet) hoặc so sánh 1 bit rồi bước cố định (QuantileNet — gần với tinh thần ngưỡng của SNN nhưng dùng bit để dịch trạng thái chứ không phát xung). Khác RNN: phép đệ quy là vô hướng theo từng đơn vị, không có ma trận bên trong vòng lặp. Khác CNN và Transformer: trạng thái có kích thước cố định, không phụ thuộc độ dài cửa sổ.

**Bộ nhớ của từng model trên cùng một tín hiệu.** S4D/Mamba lọc tuyến tính (làm mượt); HysteresisNet giữ cực trị và chỉ bám theo khi tín hiệu đi xa hơn r (dạng bậc thang, không phụ thuộc tốc độ); QuantileNet bám theo bằng bước cố định và dừng ở phân vị τ; DualMemNet có cả hai; VortexNet đếm góc cuộn quanh tâm học được; TropicalNet giữ đỉnh/đáy với độ suy giảm tuyến tính; đối chứng không bộ nhớ chỉ chép lại u.

![Bộ nhớ của từng model](docs/arch/memories.svg)

**Edge-SSM v2 (ý tưởng ban đầu).** Mỗi cạnh i → j có bộ nhớ τ_ij và hàm φ_ij riêng, nơ-ron cộng các cạnh, sau đó một lớp KAN tĩnh. Các họ mới giữ ý tưởng bộ nhớ theo đơn vị nhưng bỏ bố cục theo cạnh.

![Edge-SSM v2](docs/arch/edge_ssm.svg)

## 1. Xếp hạng trên UTD-MHAD (dataset quyết định, 8 split)

Hạng tính trên 33 model có đủ 8 split (17 model WHAR chạy lại bằng code paper-era + model chạy trong Edge_SSM).

| Hạng | Model | Loại | Params | Macro-F1 TB ± std |
|---|---|---|---|---|
| 1 | RecurrenceNet | họ mới (vòng 6) | — | 74.59 ± 13.59 |
| 2 | QuantileNet | họ mới | 20.8k | 74.29 ± 13.02 |
| 3 | DualMemNet (phân vị + trễ) | nâng cấp vòng 3 | 26.7k | 73.93 ± 13.20 |
| 4 | S4D | cũ | 27.7k | 73.79 ± 13.83 |
| 5 | HysteresisNet selective + clock | họ mới (chọn theo giao thức B) | 23.6k | 73.36 ± 12.80 |
| 6 | HysteresisNet selective + 6 widths | họ mới (chọn theo giao thức A) | 29.9k | 71.76 ± 12.78 |
| 7 | RainflowNet | họ mới (vòng 6) | — | 71.22 ± 13.68 |
| 8 | Mamba | cũ | 21.1k | 70.69 ± 13.37 |
| 9 | OrdinalNet | họ mới (vòng 6) | — | 70.49 ± 15.85 |
| 10 | HysteresisNet (base) | họ mới | 20.4k | 70.27 ± 13.23 |
| 11 | TropicalNet | đã có tiền lệ (Peng 2018) | 26.4k | 70.26 ± 12.12 |
| 12 | ReturnMapNet | họ mới (vòng 7) | — | 69.37 ± 15.68 |
| 13 | Cùng mạng, không bộ nhớ | đối chứng | 20.0k | 67.61 ± 12.06 |
| 14 | Attend+Discriminate | WHAR (paper-era) | 374.4k | 67.48 ± 13.29 |
| 15 | CNN-HAR | WHAR (paper-era) | 72.0k | 67.24 ± 13.71 |
| 21 | VortexNet | họ mới | 17.0k | 63.42 ± 10.24 |
| 27 | Edge-SSM v2 | ý tưởng ban đầu | 33.6k | 51.61 ± 11.32 |
| 29 | GRU | cũ | 40.5k | 50.14 ± 7.70 |

## 2. So sánh theo cặp (cùng split)

| Model | So với | Split | Δ TB (điểm) | Thắng / thua | Wilcoxon p |
|---|---|---|---|---|---|
| QuantileNet | S4D | cả 8 | +0.51 | 3 / 5 | 0.945 |
| QuantileNet | Mamba | cả 8 | +3.60 | 7 / 1 | 0.0781 |
| QuantileNet | Cùng mạng, không bộ nhớ | cả 8 | +6.69 | 8 / 0 | 0.00781 |
| HysteresisNet selective + clock | S4D | 4–7 (giữ lại) | +1.02 | 2 / 2 | 0.625 |
| HysteresisNet (base) | Cùng mạng, không bộ nhớ | cả 8 | +2.66 | 6 / 2 | 0.148 |

## 3. Độ bền khi tốc độ cử chỉ thay đổi (`warp_test.py`)

10 điều kiện: tốc độ ×0.5…×2 cho cả cửa sổ, và tốc độ đổi ngẫu nhiên trong cử chỉ (σ log-speed 0.25/0.5/0.75). Bảng đầy đủ và kiểm định theo cặp: `results/warp/summary_utd_mhad.md`.

| Model | Gốc ×1 | ×2 | σ 0.75 | TB 10 điều kiện | Tệ nhất |
|---|---|---|---|---|---|
| HysteresisNet selective + clock | 73.4 | 73.7 | 60.8 | 71.6 | 60.8 |
| HysteresisNet selective + 6 widths | 71.9 | 71.8 | 63.4 | 70.7 | 63.4 |
| DualMemNet (phân vị + trễ) | 73.9 | 65.6 | 57.8 | 69.3 | 57.8 |
| QuantileNet | 74.3 | 62.4 | 55.4 | 69.1 | 55.4 |
| HysteresisNet (base) | 70.3 | 70.2 | 58.5 | 68.3 | 58.5 |
| Mamba | 70.7 | 64.2 | 53.9 | 66.5 | 53.9 |
| Cùng mạng, không bộ nhớ | 67.5 | 67.3 | 50.2 | 64.5 | 50.2 |
| VortexNet | 64.1 | 61.0 | 60.8 | 63.2 | 60.8 |
| S4D | 73.8 | 33.5 | 48.2 | 57.2 | 33.5 |
| GRU | 52.0 | 44.8 | 42.1 | 48.7 | 42.1 |

## 4. Biến thể HysteresisNet và cách chọn

Giao thức A chọn theo val F1; giao thức B chọn theo test F1 split 0–3 và chỉ được đánh giá trên split 4–7.

| Biến thể | Params | Val | Test 8 split | Test 0–3 | Test 4–7 |
|---|---|---|---|---|---|
| base: 4 độ rộng, mean pooling | 20.4k | 95.79 | 70.27 | 64.13 | 76.40 |
| độ rộng chọn lọc r_t = softplus(ρ + s(x_t)) | 23.5k | 97.10 | 71.87 | 65.36 | 78.38 |
| kênh đồng hồ (vị trí tương đối trong cửa sổ) | 20.4k | 96.14 | 70.74 | 68.78 | 72.70 |
| pooling mean + max | 21.3k | 93.38 | 72.08 | 65.13 | 79.03 |
| 6 độ rộng | 26.7k | 96.47 | 71.77 | 64.05 | 79.50 |
| selective + 6 widths (giao thức A chọn) | 29.9k | 95.55 | 71.76 | 67.17 | 76.35 |
| selective + clock (giao thức B chọn) | 23.6k | 96.78 | 73.36 | 68.58 | 78.14 |
| S4D (cũ, không tinh chỉnh) | 27.7k | 95.62 | 73.79 | 70.46 | 77.11 |

## 5. Nâng cấp vòng 3 (giao thức B)

Ba biến thể chạy trên split 0–3 (chọn), chỉ biến thể được chọn chạy tiếp split 4–7 (đánh giá). Quy tắc định trước: điểm sạch dev cao nhất; nếu chênh < 1 điểm thì chọn bản bền hơn. Mỗi ô: điểm sạch / TB 10 điều kiện biến dạng / tệ nhất.

| Model | Mô tả | Dev (split 0–3) | Giữ lại (split 4–7) |
|---|---|---|---|
| DualMemNet (phân vị + trễ) | phân vị (theo thời gian) + trễ selective + clock — **được chọn trên dev** | 69.64 / 64.82 / 53.81 | 78.22 / 73.82 / 61.79 |
| dualmem_path | phân vị theo đường đi + trễ selective + clock | 64.49 / 62.73 / 55.72 | — |
| pathquantilenet | QuantileNet với bước nhảy ∝ abs(Δu) (độc lập tốc độ) | 64.18 / 61.01 / 49.02 | — |
| QuantileNet | vòng 2, cấu hình cố định trước | 67.67 / 63.31 / 49.43 | 80.91 / 74.98 / 61.29 |
| HysteresisNet selective + clock | vòng 2, chọn theo giao thức B | 68.58 / 66.86 / 56.25 | 78.14 / 76.28 / 65.27 |
| S4D | cũ | 70.46 / 54.08 / 29.00 | 77.11 / 60.35 / 37.93 |
| Mamba | cũ | 65.68 / 61.04 / 47.68 | 75.70 / 71.93 / 60.18 |

## 6. Edge-SSM v2 trên các dataset khác

| Dataset | Edge-SSM | Model WHAR tốt nhất |
|---|---|---|
| unimib_shar | 42.44 | CNN-HAR 63.92 |
| actrectut_gestures | 28.64 | Triple-Cross-Attn 61.42 |

## 7. Kiểm tra tài liệu

- **QuantileNet** (bộ nhớ = theo dõi phân vị trực tuyến, τ/η học được): chưa thấy tiền lệ làm bộ nhớ mạng sâu. Gần nhất: Behrouz et al. 2025, arXiv:2504.13173, Eq. 12 (cập nhật Sign với ℓ1, đối xứng, không τ); Ma, Muthukrishnan & Sandler 2014, arXiv:1407.1121 (Frugal Streaming, thuật toán, không học).
- **HysteresisNet** (play operator, scan song song trên monoid clamp): mới một phần. Frydrych 2026, arXiv:2605.23603 (*Preisach Attention*) đã đề xuất bộ nhớ trễ độc lập tốc độ cho sequence model — relay, chỉ lý thuyết.
- **VortexNet** (góc cuộn quanh tâm học được): chưa thấy tiền lệ là đơn vị học được; liên quan: Deep Signature Transforms (NeurIPS 2019, arXiv:1905.08494), Path Development, Polar Transformer.
- **TropicalNet** (max-plus peak-hold): **đã có** — Peng et al., *Rational Recurrences*, EMNLP 2018, arXiv:1808.09357, Phụ lục D, Eq. 24c: c_t = max{f_t + c_{t−1}, u_t} → loại.
- **Mạng dao động thích nghi tần số:** hẹp — Rohan et al. 2024, *Deep Oscillatory Neural Network*, arXiv:2405.03725; không cài đặt.
- Các phương trình của Peng (Eq. 24c) và Behrouz (Eq. 12) đã kiểm trên PDF; các bài khác kiểm trên trang tóm tắt arXiv.

## 8. Ghi chú

1. **Edge-SSM v2 (ý tưởng ban đầu) không đạt:** UTD-MHAD 51.6, UniMiB 42.4, ActRecTut 28.6. τ học được không rời phân phối khởi tạo; cố định τ cho kết quả như nhau. Dừng trước UCI-HAR (08/10).
2. **Mốc đúng là cùng mạng bỏ bộ nhớ:** MLP từng bước + mean pooling đạt 67.6 trên UTD-MHAD — đã vượt cả 17 model WHAR. Một nguyên tố bộ nhớ mới phải vượt mốc này và S4D (73.8), không phải bảng WHAR.
3. **Val F1 không dự đoán được test trên UTD-MHAD** (val lấy từ chính subject train, 93–98 %): chọn biến thể theo split 0–3, chỉ báo cáo split 4–7 (giao thức B, `select_variants.py`).
4. **Độ bền tốc độ** (`warp_test.py`): chỉ kéo giãn trục thời gian, giữ biên độ; mọi model train lại theo split và chấm trên cùng bộ biến dạng.
5. **QuantileNet với comparator cứng** (như trên MCU) giữ nguyên độ chính xác: split 0 83.5 → 84.9, split 4 76.8 → 76.0 (`check_quantile_hard.py`).
6. **Kết luận sau vòng 4 (1 dataset, 5 seed × 8 split, §9):** QuantileNet chính xác nhất (74.0), hơn Mamba có ý nghĩa (+2.1, 7/1, p = 0.023) nhưng vẫn hoà S4D (+1.45, KTC 95 % −1.5…+4.4); chưa chứng minh được không-kém-hơn-1-điểm — 8 split không đủ, cần thêm dataset (P5). Bộ nhớ play của HysteresisNet bất biến thật với tần số lấy mẫu và mất mẫu (S4D sụp), nhưng lợi thế tempo của vòng 1 bị phóng đại: khi tăng cả biên độ như cử chỉ nhanh thật thì mọi model đều sụp, HysteresisNet chỉ còn hơn S4D 9 điểm. Bộ nhớ phân vị **không** làm cả mạng bền với xung ngoại lai.
7. **Kết luận sau P5 (10 dataset × 3 seed, đăng ký trước, §10): chưa có họ mới nào hơn họ cũ.** Hạng trung bình: S4D 1.80, HysteresisNet selective + clock 2.00, QuantileNet 3.00, Mamba 3.70, không bộ nhớ 4.50 (Friedman p = 0.0004). HysteresisNet ngang S4D (5 lần hạng nhất mỗi bên, 5/5 cặp, −0.6 điểm) và hơn Mamba ở 9/10 dataset; QuantileNet dẫn ở UTD-MHAD nhưng thua S4D ở 8/10 dataset. Ở 3 dataset có baseline WHAR (ActRecTut, UniMiB, UCI-HAR), model WHAR tốt nhất (lớn hơn nhiều) vẫn hơn cả 5 model của ta → cần P6 (baseline cũ cùng kích thước).
8. **Nâng cấp vòng 3 không vượt bản vòng 2:** DualMemNet dẫn trên dev (69.6 vs QuantileNet 67.7) nhưng thua trên split giữ lại (78.2 vs 80.9); bộ theo dõi phân vị theo đường đi làm mất 3.5 điểm sạch mà không bền hơn. Với 4 split dev, nhiễu giữa các split lớn hơn khác biệt giữa kiến trúc — cần thêm seed hoặc dataset trước khi tinh chỉnh tiếp.


## 9. Vòng 4 (09/10): nhiều seed (P4), độ bền P1–P3, kiểm tra cơ chế P10

Chạy song song trên PC (GTX 1080 Ti), Colab (L4) và Kaggle (2×T4). Mỗi seed chạy cả 5 model trên cùng một máy (seed 0 và 4: PC; 1–2: Kaggle; 3: Colab). Seed huấn luyện chỉ đổi khởi tạo và thứ tự batch; split, tập val và chuẩn hoá vẫn là của giao thức (seed 0).

### P4 — nhiều seed trên UTD-MHAD

Điểm mỗi split = trung bình các seed; so sánh theo cặp trên 8 split. Bảng đầy đủ: `results/seeds/summary_utd_mhad.md`.

| Model | Seed | Macro-F1 TB | − S4D (KTC 95 %) | Thắng/thua | p | − Mamba | − không bộ nhớ |
|---|---|---|---|---|---|---|---|
| QuantileNet | 5 | 73.99 | +1.45 (-1.5…+4.4) | 4/4 | 0.461 | +2.10 | +6.87 |
| HysteresisNet selective + clock | 5 | 73.23 | +0.68 (-3.3…+4.6) | 6/2 | 0.547 | +1.34 | +6.10 |
| S4D | 5 | 72.55 | — | | | +0.66 | +5.42 |
| Mamba | 5 | 71.89 | -0.66 (-3.5…+2.2) | 2/6 | 0.461 | — | +4.77 |
| Cùng mạng, không bộ nhớ | 5 | 67.12 | -5.42 (-9.3…-1.6) | 1/7 | 0.016 | -4.77 | — |

### P1–P3 — độ bền (chấm lại checkpoint, không train lại)

Macro-F1 (%) TB trên 8 split (mỗi split TB các seed). Tempo vật lý: nhanh s lần thì gyro ×s, gia tốc động ×s², trọng lực giữ nguyên. Định nghĩa đầy đủ và kiểm định theo cặp: `robust_test.py`, `results/robust/summary_utd_mhad.md`.

| Model | Seed | Sạch | Tempo vật lý, đều (7 tốc độ) | Tempo vật lý, trong cử chỉ (3 σ) | Tần số lấy mẫu 25–100 Hz | Mất mẫu / lệch thời điểm | Nhiễu Gauss | Xung ngoại lai | Lệch offset kênh |
|---|---|---|---|---|---|---|---|---|---|
| QuantileNet | 5 | 74.0 | 48.8 | 56.8 | 68.1 | 69.1 | 64.6 | 56.8 | 31.1 |
| RecurrenceNet | 3 | 73.3 | 44.0 | 55.2 | 54.4 | 63.5 | 44.8 | 40.3 | 39.1 |
| HysteresisNet selective + clock | 5 | 73.2 | 55.2 | 57.6 | 73.3 | 72.9 | 56.7 | 40.2 | 33.1 |
| S4D | 5 | 72.5 | 45.9 | 57.5 | 52.2 | 62.1 | 66.9 | 59.2 | 27.4 |
| Mamba | 5 | 71.9 | 51.4 | 57.1 | 69.7 | 69.4 | 64.0 | 57.1 | 25.4 |
| recstatnet | 1 | 71.6 | 46.7 | 48.5 | 60.8 | 65.0 | 54.8 | 23.0 | 37.8 |
| isochron_uniform | 1 | 70.1 | 58.0 | 50.7 | 70.0 | 68.6 | 68.6 | 34.7 | 35.8 |
| statnet | 1 | 69.6 | 47.1 | 46.3 | 69.0 | 68.5 | 61.8 | 25.0 | 27.8 |
| isochron | 1 | 69.5 | 60.3 | 51.7 | 69.4 | 68.9 | 69.1 | 37.0 | 47.6 |
| Cùng mạng, không bộ nhớ | 5 | 67.1 | 46.1 | 50.4 | 67.2 | 66.2 | 57.7 | 54.4 | 20.6 |

### P9 — phân loại sớm (chấm checkpoint trên phần đầu cửa sổ)

Macro-F1 (%) sau 25/50/75/100 % cửa sổ; mọi khối đều nhân quả nên đây là đầu ra streaming tại thời điểm đó. Chi tiết: `early_test.py`, `results/early/summary_utd_mhad.md`.

| Model | 25 % | 50 % | 75 % | 100 % |
|---|---|---|---|---|
| Mamba | 17.8 | 52.6 | 67.9 | 71.9 |
| Cùng mạng, không bộ nhớ | 15.7 | 48.2 | 64.9 | 67.1 |
| RecurrenceNet | 8.4 | 44.4 | 68.9 | 73.3 |
| S4D | 11.0 | 35.7 | 62.0 | 72.5 |
| QuantileNet | 5.8 | 33.8 | 66.9 | 74.0 |
| HysteresisNet selective + clock | 8.5 | 30.1 | 65.0 | 73.2 |

### P10 — bài toán tổng hợp: mỗi bộ nhớ thắng/thua ở đâu

Độ chính xác test (%), TB ± std trên 3 seed; 4 lớp (ngẫu nhiên 25 %). "Trần" = mức tối đa một model không bộ nhớ có thể đạt vì các lớp có cùng phân phối giá trị. Chi tiết: `synthetic_test.py`, `results/synthetic/summary.md`.

**loop_dir** — chiều vẽ vòng (tam giác/vuông × xuôi/ngược), tốc độ thay đổi — dự đoán: trễ (play) thắng khi đổi tốc độ (trần không bộ nhớ 50 %).

| Model | n | iid | slow | fast | jerky |
|---|---|---|---|---|---|
| HysteresisNet selective + clock | 3 | 99.5 ± 0.3 | 95.6 ± 0.8 | 99.5 ± 0.3 | 98.6 ± 0.3 |
| HysteresisNet (base) | 3 | 99.6 ± 0.3 | 91.1 ± 0.3 | 99.8 ± 0.0 | 98.6 ± 0.8 |
| RainflowNet | 3 | 99.6 ± 0.3 | 95.4 ± 0.3 | 99.6 ± 0.5 | 97.6 ± 1.1 |
| Mamba | 3 | 99.3 ± 0.7 | 89.1 ± 5.8 | 96.1 ± 1.2 | 92.0 ± 2.0 |
| ReturnMapNet | 3 | 90.7 ± 1.2 | 84.1 ± 0.5 | 90.6 ± 2.3 | 88.8 ± 1.3 |
| QuantileNet | 3 | 99.4 ± 0.2 | 93.3 ± 2.0 | 93.7 ± 7.2 | 88.6 ± 3.3 |
| OrdinalNet | 3 | 99.9 ± 0.2 | 85.1 ± 2.6 | 49.0 ± 2.7 | 81.3 ± 1.3 |
| S4D | 3 | 98.6 ± 0.8 | 73.3 ± 2.7 | 34.6 ± 20.5 | 77.9 ± 8.9 |
| phaselocknet | 3 | 77.5 ± 4.3 | 71.4 ± 4.7 | 73.6 ± 1.4 | 71.4 ± 2.7 |
| RecurrenceNet | 3 | 99.9 ± 0.1 | 54.4 ± 4.9 | 53.3 ± 1.9 | 53.1 ± 3.0 |
| Cùng mạng, không bộ nhớ | 3 | 44.9 ± 0.3 | 40.9 ± 1.5 | 42.3 ± 1.8 | 43.2 ± 0.4 |

**step_dir** — bậc lên/xuống lớn/nhỏ với xung ngoại lai — dự đoán: phân vị thắng khi nhiều ngoại lai (trần không bộ nhớ 50 %).

| Model | n | spikes0 | spikes5 | spikes15 | spikes30 |
|---|---|---|---|---|---|
| Mamba | 3 | 100.0 ± 0.0 | 99.9 ± 0.2 | 95.3 ± 3.9 | 71.2 ± 7.5 |
| RecurrenceNet | 3 | 99.8 ± 0.2 | 100.0 ± 0.0 | 98.2 ± 0.7 | 69.4 ± 14.1 |
| RainflowNet | 3 | 95.6 ± 0.5 | 94.1 ± 0.5 | 83.5 ± 1.0 | 62.3 ± 2.0 |
| S4D | 3 | 99.6 ± 0.3 | 99.9 ± 0.2 | 95.3 ± 2.4 | 52.6 ± 3.4 |
| phaselocknet | 3 | 99.5 ± 0.2 | 99.8 ± 0.1 | 90.7 ± 3.9 | 47.8 ± 1.1 |
| HysteresisNet selective + clock | 3 | 99.7 ± 0.2 | 99.6 ± 0.3 | 85.0 ± 6.5 | 46.6 ± 0.3 |
| ReturnMapNet | 3 | 98.7 ± 0.8 | 98.9 ± 1.0 | 88.1 ± 3.7 | 46.1 ± 1.0 |
| QuantileNet | 3 | 99.4 ± 0.3 | 100.0 ± 0.0 | 79.0 ± 5.8 | 46.1 ± 0.3 |
| HysteresisNet (base) | 3 | 99.2 ± 0.5 | 98.8 ± 0.2 | 81.8 ± 6.0 | 45.9 ± 0.6 |
| OrdinalNet | 3 | 99.2 ± 0.1 | 98.3 ± 0.2 | 77.7 ± 3.5 | 40.7 ± 0.6 |
| Cùng mạng, không bộ nhớ | 3 | 42.6 ± 4.7 | 45.8 ± 2.6 | 38.7 ± 2.7 | 22.9 ± 1.3 |

**freq** — phân biệt tần số sin 1.5–3 Hz — đối chứng âm: S4D thắng, trễ thuần không làm được (trần không bộ nhớ 25 %).

| Model | n | iid | noisy |
|---|---|---|---|
| S4D | 3 | 100.0 ± 0.0 | 99.9 ± 0.1 |
| RecurrenceNet | 3 | 99.7 ± 0.3 | 98.6 ± 1.7 |
| phaselocknet | 3 | 99.4 ± 0.3 | 96.6 ± 0.6 |
| OrdinalNet | 3 | 100.0 ± 0.0 | 95.4 ± 2.3 |
| Mamba | 3 | 99.4 ± 0.2 | 76.8 ± 8.9 |
| QuantileNet | 3 | 99.7 ± 0.2 | 65.1 ± 4.8 |
| RainflowNet | 3 | 97.5 ± 1.2 | 52.2 ± 1.6 |
| HysteresisNet selective + clock | 3 | 98.8 ± 0.5 | 50.4 ± 6.5 |
| ReturnMapNet | 3 | 55.9 ± 22.6 | 29.3 ± 1.0 |
| Cùng mạng, không bộ nhớ | 3 | 25.9 ± 2.1 | 25.2 ± 1.5 |
| HysteresisNet (base) | 3 | 24.4 ± 1.3 | 24.9 ± 0.1 |

**long_order** — thứ tự dấu của hai xung cách nhau ~120 mẫu — cần nhớ xa (trần không bộ nhớ 75 %).

| Model | n | iid | distractors |
|---|---|---|---|
| S4D | 3 | 99.1 ± 0.5 | 98.4 ± 0.1 |
| QuantileNet | 3 | 99.7 ± 0.2 | 98.3 ± 0.6 |
| HysteresisNet selective + clock | 3 | 99.5 ± 0.2 | 98.1 ± 0.9 |
| phaselocknet | 3 | 99.0 ± 0.3 | 97.2 ± 1.5 |
| Mamba | 3 | 99.7 ± 0.0 | 97.2 ± 1.2 |
| ReturnMapNet | 3 | 99.2 ± 0.6 | 96.6 ± 0.4 |
| RecurrenceNet | 3 | 99.1 ± 0.9 | 95.6 ± 1.1 |
| HysteresisNet (base) | 3 | 98.8 ± 0.4 | 92.0 ± 2.0 |
| OrdinalNet | 3 | 98.2 ± 0.6 | 91.4 ± 1.2 |
| RainflowNet | 3 | 98.3 ± 1.3 | 90.3 ± 0.5 |
| Cùng mạng, không bộ nhớ | 3 | 71.2 ± 1.1 | 65.8 ± 2.8 |


## 10. P5: nhiều dataset (đang chạy / đã xong)

10 dataset WHAR chọn theo quy tắc đăng ký trước (tự tải, không lỗi dữ liệu đỏ/cam, ≤ 2.2 triệu mẫu-cửa-sổ; mhealth bị loại vì chính code paper-era không tiền xử lý được), code paper-era, seed 0–2, mỗi split TB các seed rồi TB các split. Thiết kế: `RESEARCH_PLAN.md` §5b P5; bảng đầy đủ và kiểm định: `results/p5/summary.md`.

| Dataset | Trạng thái | QuantileNet | HysteresisNet sel+clock | S4D | Mamba | No memory | RecurrenceNet |
|---|---|---|---|---|---|---|---|
| utd_mhad | đủ, 8 splits | 73.78 | 73.94 | 73.27 | 71.28 | 67.60 | 73.32 |
| actrectut_gestures | đủ, 2 splits | 36.07 | 43.57 | 43.43 | 36.70 | 36.46 | 40.37 |
| opportunity | đủ, 4 splits | 63.40 | 63.81 | 61.18 | 62.35 | 59.62 | 61.59 |
| unimib_shar | đủ, 10 splits | 59.74 | 55.50 | 60.50 | 59.11 | 50.02 | 60.78 |
| uci_har | đủ, 10 splits | 94.67 | 94.49 | 94.86 | 93.91 | 92.77 | 96.62 |
| sad | đủ, 10 splits | 85.69 | 87.59 | 87.32 | 86.33 | 84.12 | 92.93 |
| hapt | đủ, 10 splits | 82.11 | 84.33 | 83.57 | 82.37 | 81.25 | 83.32 |
| dsads | đủ, 8 splits | 81.31 | 83.04 | 84.44 | 79.60 | 79.88 | 85.55 |
| wisdm | đủ, 10 splits | 79.51 | 80.37 | 82.99 | 77.85 | 67.77 | 82.92 |
| daphnet | đủ, 10 splits | 40.71 | 40.29 | 41.36 | 39.57 | 41.11 | 42.65 |

Hạng trung bình trên 10 dataset đủ kết quả (1 = tốt nhất): RecurrenceNet 2.00, S4D 2.50, HysteresisNet sel+clock 2.60, QuantileNet 3.80, Mamba 4.60, No memory 5.50.

![Sơ đồ critical difference](results/p5/cd_recurrencenet.svg)

**Vòng 6–7 (10/10):** 5 họ mới dựa trên so sánh/sự kiện/đếm (RainflowNet, OrdinalNet, RecurrenceNet, ReturnMapNet, PhaseLockNet), sàng lọc 3 giai đoạn đăng ký trước. Chỉ **RecurrenceNet** vượt qua: hạng TB tốt nhất trên 10 dataset (2.00 so với S4D 2.50), không-kém-hơn S4D (+0.71, KTC −0.83…+2.26), hơn Mamba (+3.10); trên 6 dataset giữ lại: +1.35 so với S4D (4/6). Chi tiết: `NEW_ARCHITECTURES.md` (vòng 6–7), `results/stage2_summary.md`, `results/p5/summary_recurrencenet.md`.

## 11. So với bảng benchmark của paper WHAR Arena (Figure 6)

Không train thêm: đặt kết quả đã có vào bảng macro-F1 của paper (17 model, cùng code paper-era, cùng split; harness của ta tái lập bảng đó với ρ 0.90–0.98, lệch TB tuyệt đối 0.5–2.6 điểm/ô). Model của ta lấy seed 0 (lượt chạy protocol, giống paper chỉ chạy một lần). Hạng TB = xếp hạng trong 23 model ở từng dataset rồi lấy TB trên 10 dataset. Tham số tính trên cấu hình UTD-MHAD. Mọi trung bình ở đây chỉ trên 10 dataset của ta, dễ hơn trung bình: Leaderboard trên web lấy TB trên 30 dataset nên thấp hơn (CNN-HAR 67.7 trên 30, 73.2 trên 10 này). Bảng đầy đủ và kiểm định cặp: `results/arena/compare.md` (`compare_arena.py`).

| # | Model | Tham số | F1 TB (10) | Dev (4) | Giữ lại (6) | Bỏ ActRecTut (9) | Hạng TB /23 |
|---|---|---|---|---|---|---|---|
| 1 | TinyHAR | 185.1k | 73.13 | 67.89 | 76.63 | 75.59 | 5.60 |
| 2 | CNN-HAR | 72.0k | 73.18 | 69.94 | 75.34 | 75.51 | 6.20 |
| 3 | **RecurrenceNet** | 24.7k | 72.25 | 67.09 | 75.69 | 76.38 | 6.20 |
| 4 | DANA | 460.8k | 72.86 | 69.86 | 74.86 | 74.60 | 7.00 |
| 5 | Attend+Discriminate | 374.4k | 72.74 | 69.00 | 75.24 | 74.85 | 7.60 |
| 6 | Triple-Cross-Attn | 455.0k | 73.48 | 69.91 | 75.87 | 74.82 | 7.80 |
| 7 | TinierHAR | 8.0k | 71.39 | 64.94 | 75.68 | 74.30 | 7.95 |
| 8 | DeepConvLSTM-Attn | 477.4k | 71.20 | 66.39 | 74.40 | 73.27 | 9.70 |
| 9 | **S4D** | 27.7k | 71.25 | 68.10 | 73.35 | 74.31 | 10.30 |
| 10 | MLP-HAR | 217.2k | 69.31 | 60.20 | 75.38 | 72.15 | 11.60 |
| 11 | **HysteresisNet sel+clock** | 23.6k | 70.25 | 66.41 | 72.81 | 73.32 | 11.90 |
| 12 | **QuantileNet** | 20.8k | 68.86 | 65.76 | 70.92 | 72.59 | 12.40 |
| 13 | DeepConvLSTM | 460.8k | 68.97 | 63.12 | 72.88 | 71.28 | 12.50 |
| 14 | DynamicWHAR | 1.13M | 69.61 | 62.69 | 74.22 | 72.73 | 12.60 |
| 15 | MLP-Mixer | 621.1k | 69.21 | 63.61 | 72.95 | 72.70 | 12.90 |
| 16 | Random Forest | – | 65.30 | 55.57 | 71.79 | 67.90 | 13.40 |
| 17 | **Mamba** | 21.1k | 67.90 | 63.68 | 70.71 | 71.92 | 13.40 |
| 18 | DeepConvShallowLSTM | 402.8k | 61.83 | 48.47 | 70.74 | 63.49 | 14.90 |
| 19 | Guan-LSTM | 803.6k | 63.32 | 51.78 | 71.01 | 66.42 | 15.25 |
| 20 | **No memory (control)** | 20.0k | 65.63 | 61.95 | 68.09 | 68.72 | 15.90 |
| 21 | SA-HAR | 415.5k | 61.82 | 55.75 | 65.86 | 65.24 | 19.60 |
| 22 | SVM | – | 53.11 | 35.86 | 64.61 | 55.95 | 19.60 |
| 23 | k-NN | – | 51.83 | 39.29 | 60.18 | 54.40 | 21.70 |

Hạng nếu chèn model của ta vào 17 model của paper trên từng dataset (1 = trên tất cả), seed 0:

| Dataset | Paper tốt nhất | RecurrenceNet | S4D | HysteresisNet sel+clock | QuantileNet | Mamba | No memory (control) |
|---|---|---|---|---|---|---|---|
| utd_mhad | 68.2 CNN-HAR | 74.6 (#1) | 73.8 (#1) | 73.4 (#1) | 74.3 (#1) | 70.7 (#1) | 67.6 (#2) |
| uci_har | 96.1 TinyHAR | 96.7 (#1) | 94.8 (#6) | 94.3 (#7) | 94.7 (#6) | 93.5 (#8) | 92.7 (#10) |
| unimib_shar | 63.3 CNN-HAR | 62.0 (#3) | 60.1 (#5) | 55.3 (#10) | 58.8 (#5) | 58.8 (#5) | 49.6 (#12) |
| actrectut_gestures | 61.5 Triple-Cross-Attn | 35.1 (#15) | 43.8 (#11) | 42.7 (#11) | 35.2 (#15) | 31.7 (#15) | 37.9 (#13) |
| opportunity * | 74.7 MLP-HAR | 64.9 (#10) | 61.0 (#14) | 64.6 (#10) | 60.9 (#14) | 61.9 (#13) | 57.8 (#18) |
| sad * | 98.2 Random Forest | 92.0 (#10) | 89.4 (#16) | 87.1 (#16) | 82.3 (#16) | 82.8 (#16) | 80.8 (#18) |
| hapt * | 84.8 CNN-HAR | 83.2 (#3) | 82.1 (#3) | 84.4 (#2) | 82.7 (#3) | 82.8 (#3) | 81.1 (#6) |
| dsads * | 89.3 Random Forest | 86.2 (#6) | 85.2 (#9) | 80.1 (#15) | 80.9 (#15) | 80.8 (#15) | 79.1 (#15) |
| wisdm * | 85.5 DANA | 84.3 (#4) | 82.5 (#11) | 80.9 (#13) | 78.4 (#13) | 76.5 (#13) | 69.1 (#15) |
| daphnet * | 45.4 TinierHAR | 43.5 (#4) | 39.8 (#10) | 39.7 (#10) | 40.2 (#10) | 39.4 (#10) | 40.7 (#7) |

\* dataset giữ lại (không dùng để chọn gì).

**Đọc bảng:** RecurrenceNet (24.7k tham số) có hạng TB thứ 3/23, ngang CNN-HAR (72k) và sau TinyHAR (185k). F1 TB 72.25 thấp hơn nhóm đầu 0.5–1.2 điểm, nhưng không cặp nào khác biệt có ý nghĩa (10 dataset, Wilcoxon + Holm). Phần thua dồn vào ActRecTut (2 split, 35 so với 61.5). Bỏ dataset này thì RecurrenceNet có F1 TB cao nhất (76.38 so với TinyHAR 75.59), nhưng đó là phân tích độ nhạy đặt ra sau khi xem số, không phải kết luận. Trên UTD-MHAD mọi model có bộ nhớ của ta vượt cả 17 model của paper, song đối chứng không bộ nhớ đã đạt 67.6 (CNN-HAR 68.2) và UTD-MHAD là dataset phát triển, nên phần lớn lợi thế đến từ khung chung. Điểm yếu rõ: SAD và DSADS (Random Forest với đặc trưng tsfresh thắng), Opportunity (MLP-HAR 74.7), ActRecTut.

## 12. Tổng hợp vòng 1–13 và các yếu tố ảnh hưởng (nhận định ngày 12/10)

Số liệu: `synthesis.py` → `results/synthesis/summary.md` (không train thêm). Dùng TB 3 seed nếu có, ngược lại seed 0. "Probe" là thống kê đoạn + bộ phân loại cổ điển (`tree_probe.py`); không chạy trên DSADS và Daphnet. 10 dataset này là tập phát triển, nên mọi p-value đều lạc quan.

### 12.1 Bảng tổng (macro-F1 %)

| Model | UTD | UCI | UniMiB | ActRecTut | Opp. | SAD | HAPT | DSADS | WISDM | Daphnet | TB 10 | TB bỏ ActRecTut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RN + thống kê kênh (recstatnet) | 72.2 | 96.3 | 61.8 | 61.7 | 63.9 | 95.9 | 84.5 | 86.1 | 81.9 | 42.8 | 74.70 | 76.15 |
| RN + LawNet (lawnet_rec) | 72.9 | 96.7 | 64.4 | 48.4 | 62.5 | 98.8 | 84.2 | 89.8 | 82.7 | 43.8 | 74.42 | 77.31 |
| RecurrenceNet (RN) | 73.3 | 96.6 | 60.8 | 40.4 | 61.6 | 92.9 | 83.3 | 85.6 | 82.9 | 42.7 | 72.01 | 75.52 |
| S4D | 73.3 | 94.9 | 60.5 | 43.4 | 61.2 | 87.3 | 83.6 | 84.4 | 83.0 | 41.4 | 71.29 | 74.39 |
| HysteresisNet sel+clock | 73.9 | 94.5 | 55.5 | 43.6 | 63.8 | 87.6 | 84.3 | 83.0 | 80.4 | 40.3 | 70.69 | 73.70 |
| LawNet (không bộ nhớ + luật) | 70.7 | 94.6 | 58.6 | 43.5 | 61.1 | 96.7 | 81.5 | 87.3 | 78.2 | 43.2 | 71.52 | 74.63 |
| Thống kê (không bộ nhớ + thống kê) | 67.3 | 92.6 | 54.1 | 55.6 | 62.0 | 93.1 | 81.5 | 83.2 | 71.5 | 41.5 | 70.24 | 71.86 |
| QuantileNet | 73.8 | 94.7 | 59.7 | 36.1 | 63.4 | 85.7 | 82.1 | 81.3 | 79.5 | 40.7 | 69.70 | 73.43 |
| Mamba | 71.3 | 93.9 | 59.1 | 36.7 | 62.3 | 86.3 | 82.4 | 79.6 | 77.9 | 39.6 | 68.91 | 72.48 |
| Không bộ nhớ (đối chứng) | 67.6 | 92.8 | 50.0 | 36.5 | 59.6 | 84.1 | 81.3 | 79.9 | 67.8 | 41.1 | 66.06 | 69.35 |
| StyleAlignNet (seed 0) | 75.2 | 96.2 | 59.3 | 48.7 | 65.8 | 93.5 | 83.3 | 88.0 | 83.9 | 43.5 | 73.74 | 76.52 |
| Probe: thống kê cửa sổ + ExtraTrees | 61.8 | 90.6 | 50.8 | 65.6 | 72.0 | 98.7 | 80.9 | – | 76.4 | – | – | – |
| Probe: thống kê 31 đoạn + ExtraTrees | 72.2 | 89.8 | 57.3 | 49.4 | 72.5 | 97.4 | 80.7 | – | 75.8 | – | – | – |
| 1-NN DTW chuẩn hoá | 74.3 | 86.1 | 58.4 | 69.3 | 31.7 | 88.0 | 71.6 | 77.3 | 63.5 | 32.5 | 65.26 | 64.81 |
| Paper: tốt nhất từng dataset | 68.2 | 96.1 | 63.3 | 61.5 | 74.7 | 98.2 | 84.8 | 89.3 | 85.5 | 45.4 | 76.70 | 78.40 |
| Paper: CNN-HAR (72k) | 68.2 | 96.0 | 63.3 | 52.2 | 63.1 | 95.6 | 84.8 | 86.9 | 81.4 | 40.2 | 73.18 | 75.51 |
| Paper: TinyHAR (185k) | 65.8 | 96.1 | 58.6 | 51.1 | 65.4 | 97.7 | 81.0 | 86.8 | 85.0 | 43.9 | 73.13 | 75.59 |
| Paper: Random Forest | 51.2 | 84.7 | 44.5 | 41.9 | 70.6 | 98.2 | 67.1 | 89.3 | 71.6 | 33.8 | 65.30 | 67.90 |

So với từng model của benchmark (17 model của paper), ghép cặp trên 10 dataset ở seed 0: hai biến thể tốt nhất hơn mọi model của paper về trung bình, nhưng với 8 model mạnh nhất thì khoảng tin cậy 95 % đều chứa 0 và không phép kiểm định nào (Holm) có ý nghĩa; thắng rõ chỉ với các model sâu yếu hơn và các model cổ điển. Bảng: `results/vs_benchmark/summary.md` (`vs_benchmark.py`).

![So sánh trực tiếp với từng model của benchmark](docs/factors/vs_benchmark.svg)


### 12.2 Các vòng và kết luận

- Vòng 1–3 (chỉ UTD-MHAD): Edge-SSM v2 thất bại (51.6). HysteresisNet và QuantileNet ngang S4D trên UTD.
- Vòng 4 (5 seed trên UTD): QuantileNet 74.0 so với S4D 72.5, không có ý nghĩa. Bộ nhớ trễ bền với tần số lấy mẫu.
- Vòng 5 (P5, 10 dataset × 3 seed): không họ mới nào thắng S4D (hạng TB S4D 1.80).
- Vòng 6–7: chỉ RecurrenceNet vượt qua. TB 72.0 so với S4D 71.3 (+0.71, KTC −0.83…+2.26), không có ý nghĩa.
- Vòng 8 (ghép đặc trưng cấp cửa sổ vào khung): recstatnet 74.70 và lawnet_rec 74.42 (3 seed). lawnet_rec − S4D +3.12 [+0.58, +5.67], Holm p 0.049: kết quả duy nhất có ý nghĩa so với S4D (trên tập phát triển).
- Vòng 9–10: năm họ "táo bạo" (ParliamentNet, ForestNet, EcoNet, GrammarNet, braid-holonomy) đều dừng. Mẫu chung: một bộ nhớ mới trong cùng khung nhỏ luôn nằm giữa đối chứng không bộ nhớ và RN.
- Vòng 11: phân tích lỗi (dư địa lớn nhất: phân biệt kiểu dáng đi, lớp null). 1-NN DTW chuẩn hoá đứng #1 trên cử chỉ và kém nhất trên vận động. StyleAlignNet 73.74 (seed 0), nhưng đối chứng cho thấy lợi ích đến từ đặc trưng cửa sổ, không từ phần căn chỉnh.
- Vòng 11b: ablation (đòn bẩy là biên độ: std, min/max; mean vô ích). Test a/b nghiêng về H2: thông tin biên độ theo kênh phải lấy trước khi trộn kênh. 3 seed xác nhận thứ tự của seed 0.
- Vòng 12 (session khác): IsochronNet dừng (−2.7 và −2.3 so với recstatnet). Pha nội tại giúp dáng đi: WISDM +5.2 so với chia thời gian đều, p 0.049, một seed.
- §5k (session khác): lợi thế "S4D rơi xuống 44 ở 25 Hz" chỉ đúng với pipeline ngây thơ. Khi nội suy về lưới hoặc đổi Δ, S4D giữ 73.4 ở ×0.5 tần số.
- Vòng 13: thống kê đoạn + bộ phân loại cổ điển (không học sâu) ≈ RN/S4D trên TB 8 dataset. Ba ứng viên cấu trúc mới đều "chỉ mới một phần" (QUANT, Koopa, DDNN) và không được xây.

### 12.3 Mức lợi của từng thành phần trên từng dataset (điểm macro-F1)

| Đối chiếu | UTD | UCI | UniMiB | ActRecTut | Opp. | SAD | HAPT | DSADS | WISDM | Daphnet | TB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Có bộ nhớ: RN − không bộ nhớ | +5.7 | +3.9 | +10.8 | +3.9 | +2.0 | +8.8 | +2.1 | +5.7 | +15.2 | +1.5 | +5.95 |
| Có bộ nhớ: S4D − không bộ nhớ | +5.7 | +2.1 | +10.5 | +7.0 | +1.6 | +3.2 | +2.3 | +4.6 | +15.2 | +0.2 | +5.23 |
| Thống kê trên mạng không bộ nhớ | -0.3 | -0.2 | +4.1 | +19.2 | +2.4 | +9.0 | +0.2 | +3.3 | +3.7 | +0.4 | +4.18 |
| Luật trên mạng không bộ nhớ | +3.1 | +1.8 | +8.6 | +7.0 | +1.4 | +12.5 | +0.2 | +7.4 | +10.4 | +2.0 | +5.46 |
| Thống kê thêm vào RN | -1.1 | -0.3 | +1.0 | +21.4 | +2.3 | +3.0 | +1.2 | +0.6 | -1.0 | +0.1 | +2.70 |
| Luật thêm vào RN | -0.4 | +0.1 | +3.6 | +8.0 | +0.9 | +5.8 | +0.9 | +4.2 | -0.3 | +1.2 | +2.41 |
| Cổ điển (31 đoạn + ET) − RN | -1.2 | -6.9 | -3.4 | +9.0 | +10.9 | +4.4 | -2.6 | – | -7.1 | – | +0.39 |
| 1-NN DTW chuẩn hoá − RN | +1.0 | -10.6 | -2.4 | +29.0 | -29.9 | -5.0 | -11.7 | -8.3 | -19.4 | -10.1 | -6.74 |
| RN − S4D | +0.0 | +1.8 | +0.3 | -3.1 | +0.4 | +5.6 | -0.3 | +1.1 | -0.1 | +1.3 | +0.71 |
| S4D − Mamba | +2.0 | +0.9 | +1.4 | +6.7 | -1.2 | +1.0 | +1.2 | +4.8 | +5.1 | +1.8 | +2.39 |

Chi tiết: `results/synthesis/summary.md`, `results/factors/summary.md`. Toàn bộ hình và thẻ từng họ model: `Model atlas.html`.

![Hiệu ứng của từng thành phần, từng dataset](docs/factors/components_heatmap.svg)


### 12.4 Yếu tố phía model

Sơ đồ tổng: yếu tố dữ liệu (trái), thành phần model (phải) và yếu tố đánh giá (dưới) cùng tác động lên macro-F1 trên người chưa thấy. Độ dày mũi tên theo độ lớn hiệu ứng đo được; nét đứt là không có ý nghĩa hoặc đổi dấu giữa các dataset.

![Sơ đồ các yếu tố ảnh hưởng tới macro-F1](docs/factors/factor_map.svg)

So sánh có kiểm soát giữa các model của ta, ghép cặp trên cùng split (TB 3 seed), khoảng tin cậy 95 % qua 10 dataset:

![Hiệu ứng ghép cặp của từng thành phần model](docs/factors/components_forest.svg)

- **Có bộ nhớ thời gian hay không — yếu tố lớn nhất phía model.** RN − không bộ nhớ +5.95 (S4D +5.23), dương trên cả 10 dataset. Lợi lớn nhất ở WISDM +15.2, UniMiB +10.8, SAD +8.8, UTD +5.7; nhỏ ở Daphnet +1.5, Opportunity +2.0, HAPT +2.1. Lợi đi cùng nhiều lớp hơn (ρ +0.48) và ít kênh hơn (−0.45), nhưng không gắn với số mẫu mỗi cửa sổ (ρ −0.36). Đây là mô tả, chưa có ý nghĩa thống kê.
- **Loại bộ nhớ — nhỏ hơn nhiều so với có hay không.** Giữa các bộ nhớ chỉ chênh ≤ 3 điểm TB: RN 72.0, S4D 71.3, HysteresisNet 70.7, QuantileNet 69.7, Mamba 68.9. RN − S4D +0.71, chủ yếu do SAD +5.6. Mamba thấp nhất: S4D − Mamba +2.39, S4D hơn trên 9/10 dataset, nên tính chọn lọc theo đầu vào không có lợi ở cỡ dữ liệu này. Cơ chế thắng trên bài tổng hợp (P10) không chuyển sang dữ liệu thật.
- **Đặc trưng cấp cửa sổ ghép muộn (thống kê kênh, "luật").**
  - Trên mạng không bộ nhớ: thống kê +4.18, luật +5.46. Luật mang thông tin thời gian qua tự tương quan ở nhiều độ trễ (WISDM +10.4, SAD +12.5), nên gần như thay được bộ nhớ.
  - Thêm vào RN: +2.70 và +2.41. Lợi của thống kê gần như chỉ ở ActRecTut (+21.4; 9 dataset còn lại +0.6), và nó mất ~1 điểm trên UTD. Lợi của luật đều hơn: +1.8 khi bỏ ActRecTut; SAD +5.8, DSADS +4.2, UniMiB +3.6.
  - Phần có tác dụng là biên độ (std, min/max), không phải mean. Nhóm kênh ngẫu nhiên tốt ngang cảm biến thật (72.1 so với 71.7, seed 0). Lợi khi thêm vào RN tăng theo số kênh (ρ +0.57 / +0.45).
  - Kiểm tra tài liệu lại ngày 12/10 (đọc nguyên văn): nhánh thống kê theo từng trục (có mean, std, min, max) ghép với đặc trưng sâu trước lớp phân loại, huấn luyện chung, đã có ở Ravì et al. (IEEE JBHI 2017), cũng nhắm tới thiết bị biên. DDNN (IJCAI 2019) chỉ giống ở khuôn: nhánh thống kê của nó được học (autoencoder + MMD), có 3 nhánh.
  - Nhánh luật: đặc trưng toán tử tuyến tính ghép với mạng sâu đã có (Dong et al. 2021: DMD trên khung xương từ video, không end-to-end). Chưa thấy: toán tử trễ khớp theo từng cửa sổ trên IMU, đọc qua bất biến GL(3) (đa thức đặc trưng), nhiều độ trễ, tương quan chính tắc có độ trễ giữa các cảm biến, huấn luyện end-to-end.
- **Chia sẻ trọng số theo cảm biến (ParliamentNet, seed 0).** SAD 96.9 và Opportunity 70.7 (+4.9 / +5.8 so với RN), nhưng UTD 58.6 (−16). Có lợi khi có nhiều cảm biến cùng loại, có hại khi chỉ có một bo mạch.
- **Bất biến co giãn và tempo.** 1-NN DTW chuẩn hoá thắng cử chỉ (ActRecTut 69.3, UTD 74.3), nhưng thua nặng trên vận động (Opportunity 31.7, WISDM 63.5; TB −6.7 so với RN). Chuẩn hoá theo cửa sổ xoá mất tư thế và cường độ. Pha nội tại (IsochronNet) giúp dáng đi nhưng không giúp cử chỉ. Phiên bản học được (StyleAlignNet) không tái tạo được DTW.
- **Chuẩn hoá và pooling bên trong khối (test a/b).** Chỉ lấy lại ≤ 2.7 trong 6.4 điểm. Bỏ LayerNorm làm SAD giảm 2.7 điểm.
- **Kích thước model không quyết định.** Trong bảng paper, log(tham số) và F1 TB có ρ −0.37: lớn hơn không tốt hơn. TinierHAR (8k) đạt 71.4. Các model 25–27k của ta đạt 72–75, ngang hoặc hơn các model 70k–1.1M.
- **Độ bền (UTD, P1–P3): không model nào bền với mọi kiểu nhiễu.** S4D tốt nhất với nhiễu Gauss (66.9) và xung (59.2). HysteresisNet tốt nhất với tempo vật lý (55.2) và tần số lấy mẫu (73.3). RN kém với nhiễu (44.8). Lệch offset làm mọi model sụp (20–39), trừ IsochronNet (47.6). Lợi thế về tần số lấy mẫu mất khi baseline có pipeline chuẩn.

### 12.5 Yếu tố phía dữ liệu

![Yếu tố dữ liệu trên 30 dataset của paper](docs/factors/data_factors.svg)

![Tương quan và hồi quy giữa thuộc tính dataset và kết quả](docs/factors/data_corr.svg)

- **Trên 30 dataset của paper** (17 model; `factor_analysis.py`):
  - Không thuộc tính đơn giản nào dự đoán được độ khó của dataset: số người, số kênh, số lớp và độ dài cửa sổ cùng lúc cho R² 0.03 với F1 trung vị của các model sâu.
  - Thuộc tính quyết định **loại model nào thắng**. Nhiều kênh hơn thì RF tiến gần hoặc vượt model sâu: +7.2 điểm mỗi sd, p < 0.001, R² 0.56. Ít người hơn thì chọn model nào quan trọng hơn: best − median −1.5 điểm mỗi sd, p 0.005.
  - Nhiều kênh hơn thì kết quả phụ thuộc vào người test nhiều hơn: split sd +2.5 điểm mỗi sd, p 0.002.
  - Số người và số kênh tương quan với nhau (ρ −0.50), nên chỉ tách được một phần.
- **Số người huấn luyện quyết định "cổ điển hay học sâu".** Hiệu (31 đoạn + ET) − RN tương quan với log2 số người ρ −0.93 (n = 8).
  - Ít người (ActRecTut 2, Opportunity 4): thống kê + cây thắng (+9, +11).
  - Nhiều người (UCI, HAPT, UniMiB 30; WISDM 36): mạng sâu thắng (3–7 điểm).
  - Mạng sâu cần đủ nhiều người khác nhau để học được đặc trưng thời gian tổng quát.
- **Số kênh có cùng chiều (ρ +0.81)**, nhưng trong 8 dataset này nó lẫn với số người (Opportunity 133 kênh / 4 người, SAD 60 / 10). Trên 30 dataset của paper, hiệu RF − mô hình sâu tốt nhất tăng theo số kênh (ρ 0.71), và vẫn đúng sau khi kiểm soát các biến khác (partial 0.73).
- **Loại hoạt động.**
  - Cử chỉ (UTD, ActRecTut): hình dạng, thứ tự và phong cách từng người; DTW và thống kê mạnh.
  - Dáng đi và vận động lặp (WISDM, UniMiB, SAD): bộ nhớ thời gian mạnh nhất.
  - Nhiều cảm biến (Opportunity, SAD, DSADS): thống kê hoặc luật theo nhóm kênh, và chia sẻ trọng số.
  - Daphnet: mọi model chỉ 40–45 vì trần metric là 66.7 (lớp vắng tính bằng 0), gần như không phân biệt được.
- **Độ dài cửa sổ và tần số lấy mẫu.** Protocol dùng cửa sổ 3 s cho mọi dataset (ghi đè cấu hình của từng dataset). Vì vậy số mẫu mỗi cửa sổ (60–192) chỉ phản ánh tần số lấy mẫu. Nó không đi cùng mức lợi của bộ nhớ (ρ −0.36; WISDM 60 mẫu nhưng lợi +15), cũng không đi cùng kết quả nào trên 30 dataset (|ρ| ≤ 0.19).
- **Số lớp và cách tính macro-F1.** UTD có 27 lớp với 8 người, nên mỗi split thiếu lớp (trần split-01 là 51.9). Lớp vắng tính F1 = 0 kéo trần xuống và làm khác biệt giữa model nhỏ đi.

### 12.6 Yếu tố phía đánh giá (vì sao kết luận có thể đảo)

![Phân rã phương sai của macro-F1](docs/factors/variance.svg)

![Độ lệch theo người, theo seed và theo loại bộ nhớ; ảnh hưởng của ActRecTut](docs/factors/eval_noise.svg)

- **Phân rã phương sai của 2,460 lượt chạy** (10 model × 10 dataset × split × 3 seed):
  - Riêng dataset chiếm 76.6 % tổng phương sai.
  - Trong một dataset: người test 61 %, seed 14 %, model × người 11 %, model × dataset 7 %, model chỉ 6 %.
- **Người test chiếm phần lớn biến thiên.**
  - Độ lệch theo split (± trong paper): TB 8.1 điểm với RN; Opportunity 15.5, UTD 13.6.
  - Độ lệch theo seed chỉ 1.2–1.7 điểm (ActRecTut 6.1, Opportunity 3.4).
  - Người đã thấy − người mới (val − test): TB 14.5 điểm (ActRecTut 35.3, UTD 21.9).
  - Nên bài toán thực chất là tổng quát sang người mới, và chênh dưới ~1.5 điểm trên một dataset là nhiễu.
- **ActRecTut (2 người, 2 split) có thể đảo kết luận.** Lợi +21.4 của recstatnet nằm ở đây. Bỏ nó thì thứ tự đổi: lawnet_rec 77.31 > recstatnet 76.15. Luôn báo cáo cả có và không có ActRecTut. Kill set 4 dataset cũng bị nó chi phối.
- **Tập phát triển và tập giữ lại.** Mọi model đều được chọn trên 10 dataset này, nên p-value lạc quan. UTD-MHAD (dataset chọn ban đầu) cho ứng viên lợi thế: QuantileNet dẫn trên UTD nhưng thua S4D trên 8/10 dataset.
- **Baseline yếu làm phồng thành tích.**
  - RF của paper (tsfresh tối giản, cây sâu tối đa 8) đạt 65.3. Cùng họ đó, ExtraTrees trên thống kê đoạn đạt ~74.5 trên 8 dataset.
  - Pipeline ngây thơ cho "S4D 44 ở 25 Hz"; với pipeline chuẩn là 73.4.
- **Phần cứng và cache.**
  - T4 / L4 / 1080 Ti lệch ~1 điểm mỗi ô.
  - Cache cục bộ của DSADS và Daphnet đánh số người khác (thứ tự listdir), nên chỉ so được trung bình.
  - Kiểm định phải ghép cặp trên cùng split.
- **Một seed so với ba seed.** Thứ hạng ở seed 0 giữ nguyên qua 3 seed với các model chính (recstatnet 74.69 → 74.70).

### 12.7 Độ chính xác theo chi phí (Pareto)

Chi phí đo trên đúng đầu vào của từng dataset (cửa sổ 3 s), lấy trung bình nhân qua 10 dataset; độ chính xác là TB macro-F1 10 dataset ở seed 0 như paper. Đường liền: mặt Pareto của mọi model; đường đứt: chỉ các model của paper. Bảng: `results/pareto/summary.md` (`pareto.py`).

![Mặt Pareto: độ chính xác theo số tham số và độ trễ](docs/factors/pareto.svg)


### 12.8 Nhận định chung

Cấu trúc của họ tốt nhất hiện có (thân RecurrenceNet + nhánh cấp cửa sổ, ghép ở head):

![Cấu trúc họ model tốt nhất](docs/factors/best_family.svg)

![Mọi yếu tố trên cùng một thang](docs/factors/ranking.svg)

- Thứ tự ảnh hưởng (ước lượng thô, điểm macro-F1):
  1. dataset nào (76.6 % tổng phương sai), rồi người test (61 % phương sai trong một dataset, ± 8 điểm, khoảng cách người mới 14.5); hai yếu tố này lớn hơn hẳn mọi lựa chọn kiến trúc;
  2. có bộ nhớ thời gian (~6);
  3. đặc trưng cấp cửa sổ ghép muộn (~2.5, tập trung ở dataset ít người / nhiều kênh);
  4. loại bộ nhớ (~1–3), ngang nhiễu seed (~1.5);
  5. kích thước model: không đơn điệu.
- Hai loại thông tin bổ sung cho nhau:
  - thống kê không thứ tự tổng quát tốt sang người mới khi ít người và nhiều kênh;
  - đặc trưng thời gian thắng khi nhiều người, với dáng đi và hình dạng chi tiết.
  - Model tốt nhất (recstatnet, lawnet_rec) có cả hai. Khuôn ghép thống kê với thân sâu đã có (Ravì et al. 2017); phần chỉ mới một phần là thân RecurrenceNet (gần SELFY 2021 và Jeong & Yoo 2026) và nhánh luật bất biến GL(3).
- Chưa có họ cấu trúc mới nào thắng. Thay ô bộ nhớ trong cùng khung không đủ (vòng 1–10). Các cấu trúc mới thử ở vòng 12–13 chưa cho kết quả tốt hơn.
- Để một tuyên bố đứng vững cần:
  - baseline cổ điển mạnh (P6);
  - dataset giữ lại mới;
  - so sánh ghép cặp theo split;
  - báo cáo cả có và không có ActRecTut.

