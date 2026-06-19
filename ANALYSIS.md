# Phân tích kết quả — Day 17: Memory Systems for AI Agent

Tất cả số liệu dưới đây lấy từ `python src/benchmark.py` chạy ở chế độ **offline deterministic**
(không cần API key, kết quả tái lập 100%).

## Bảng kết quả

### Standard Benchmark (`data/conversations.json` — 10 phiên hội thoại ngắn)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline | 1176              | 12436                   | 0.0                  | 0.30             | 0                     | 0           |
| Advanced | 1662              | 19480                   | 1.0                  | 1.00             | 280                   | 0           |

### Long-Context Stress Benchmark (`data/advanced_long_context.json` — 1 thread cực dài)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline | 1021              | 21850                   | 0.0                  | 0.30             | 0                     | 0           |
| Advanced | 545               | 11241                   | 1.0                  | 1.00             | 207                   | 28          |

## 1. Vì sao Advanced recall tốt hơn Baseline?

Baseline chỉ có **short-term memory trong cùng `thread_id`**. Khi câu hỏi recall được đặt ở một
thread mới (đo cross-session đúng nghĩa), baseline không còn lịch sử nào để tra → recall = **0.0**
ở cả hai bộ dữ liệu.

Advanced có thêm **persistent memory** qua `User.md` (`UserProfileStore`). Mỗi lượt, các fact ổn định
(tên, nơi ở, nghề nghiệp, style, đồ uống, món ăn, thú cưng, mối quan tâm) được trích bằng
`extract_profile_updates()` rồi `upsert_fact()` vào file. File này sống xuyên thread, nên ở thread mới
agent vẫn đọc lại được → recall = **1.0**.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Ở Standard Benchmark, `Prompt tokens processed` của Advanced (**19480**) **cao hơn** Baseline (**12436**).
Lý do: mỗi lượt Advanced phải kéo theo thêm ngữ cảnh của `User.md` (profile) cộng với summary, trong khi
các thread ngắn **chưa đủ dài để chạm ngưỡng compact** (`Compactions = 0`). Lúc này lớp memory chỉ thêm
overhead mà chưa có cơ hội tiết kiệm. Đây là kết luận quan trọng: **memory phức tạp không miễn phí**, và
ở quy mô nhỏ nó có thể lỗ về token.

## 3. Vì sao compact giúp Advanced thắng ở hội thoại dài?

Ở Stress Benchmark, tình thế đảo ngược:

- Baseline `Prompt tokens processed` = **21850** vì nó **nạp lại toàn bộ lịch sử thread mỗi lượt** →
  chi phí cộng dồn tăng gần như bậc hai theo độ dài hội thoại.
- Advanced chỉ = **11241** (giảm ~49%) vì `_estimate_prompt_context_tokens()` chỉ tính
  `User.md + summary + vài message gần nhất`, **bị chặn trên bởi ngưỡng compact**. Trong run này compact
  đã kích hoạt **28 lần** (`CompactMemoryManager`), liên tục nén lịch sử cũ thành summary ngắn.

Điểm cần nhấn: compact tối ưu **chủ yếu ở `Prompt tokens processed`** (ngữ cảnh phải mang theo), chứ
không phải ở `Agent tokens only` (token sinh ra trong câu trả lời). Đó là lý do cột `Agent tokens only`
của hai agent không chênh nhiều, nhưng cột prompt thì khác biệt rõ rệt.

## 4. Memory file tăng trưởng ra sao và rủi ro gì?

Cột `Memory growth (bytes)` cho thấy `User.md` của Advanced phình ra (280 và 207 bytes) trong khi
Baseline luôn = 0 (không lưu gì). Đây là **chi phí thật**:

- File lớn dần theo thời gian → mỗi lượt nạp profile cũng tốn token hơn.
- Nếu trích sai (lưu nhiễu, lưu câu hỏi như thể là fact), file vừa to vừa **sai**, kéo chất lượng xuống.
- Cần guardrail để biết **cái gì đáng lưu dài hạn** và **cái gì chỉ là tạm thời**.

## 5. Phần bonus đã làm

### a) Question guardrail — không lưu fact từ câu hỏi
**Giải quyết:** tránh lỗi kinh điển khi agent biến *"Mình tên là gì?"* thành fact `name = "gì"`.
**Cách làm:** `extract_profile_updates()` bỏ qua message có `?`, có `nhắc lại giúp`, và loại giá trị
nghi vấn (`gì/nào/đâu...`); tên phải là danh từ riêng (viết hoa).
**Cải thiện:** giữ `User.md` sạch, recall không bị nhiễu. **Rủi ro:** heuristic có thể bỏ sót fact nằm
trong câu pha lẫn câu hỏi.

### b) Conflict handling — đính chính ghi đè, không giữ fact cũ sai
**Giải quyết:** dataset có nhiều correction (Đà Nẵng→Huế→Đà Nẵng, backend→MLOps engineer).
**Cách làm:** `_strip_negations()` xóa mệnh đề bị phủ định (`chứ không còn...`, `không còn... nữa`) để
lấy đúng giá trị mới; `upsert_fact()` ghi đè theo key (last-write-wins). Test
`test_conflict_handling_keeps_only_latest_fact` xác nhận `backend engineer` biến mất khỏi profile.
**Cải thiện:** recall đúng theo thông tin mới nhất. **Rủi ro:** nếu user nói mỉa/giả định, ghi đè máy móc
có thể sai → nên thêm confidence threshold.

### c) Noise filtering — bỏ qua thông tin gây nhiễu
**Giải quyết:** stress test cố tình chèn nhiễu (*"đùa product manager"*, *"Hà Nội chỉ là nơi bay ra họp"*).
**Cách làm:** bỏ câu chứa marker nhiễu (`đùa`, `bay ra họp`, `chỉ là nơi`...).
**Cải thiện:** nghề nghiệp giữ đúng `MLOps engineer`, nơi ở giữ đúng `Đà Nẵng`. **Rủi ro:** danh sách
marker cứng, khó tổng quát sang dữ liệu khác.

## 6. Câu chuyện tổng thể

1. Baseline không nhớ dài hạn → recall 0.
2. Advanced thêm `User.md` → recall 1.0.
3. Hội thoại dài làm prompt cost của baseline phình mạnh (21850).
4. Compact memory kéo chi phí ngữ cảnh của advanced xuống (11241, 28 compactions).
5. Hệ thống mạnh hơn nhưng phức tạp hơn: tốn hơn ở thread ngắn, file memory là chi phí thật, và cần
   guardrail (question/conflict/noise) để không lưu sai.
