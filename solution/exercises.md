# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng — đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu kém mạch lạc?** (2–3 câu)
> *Ở những mức temperature thấp (gần với 0), câu trả lời rất logic, chặt chẽ và ổn định, câu trả lời sẽ thiên về thông tin chính xác và hầu như lặp lại nếu chạy nhiều lần. Ở mức temperature 0.7, câu trả lời vẫn mạch lạc nhưng diễn đạt đa dạng và tự nhiên hơn. Ở mức temperature 1.2, câu trả lời trở nên linh hoạt hơn nhưng có thể mất đi độ chính xác. Ở mức temperature 1.8, câu trả lời trở nên rất ngẫu nhiên và không ổn định. Điều đó cho thấy quy luật rằng, càng tăng temperature thì độ sáng tạo của phản hồi càng cao, cùng với đó câu trả lời sẽ trở nên kém mạch lạc.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý, và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Tôi sẽ đặt temperature là 0.0 cho trợ lý soạn thảo hợp đồng pháp lý vì cần độ chính xác cao và không muốn có sự ngẫu nhiên trong các câu trả lời. Trong khi đó, tôi sẽ đặt temperature là 1.0 cho trợ lý viết slogan quảng cáo để tạo ra các ý tưởng sáng tạo và khác biệt.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần, mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này (dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Nếu sử dụng model lớn, chi phí mỗi ngày sẽ cao hơn đáng kể so với model nhỏ. Tuy nhiên, nếu ứng dụng yêu cầu độ chính xác cao và khả năng xử lý phức tạp, thì model lớn có thể xứng đáng với chi phí. Trong khi đó, nếu ứng dụng chỉ cần các chức năng cơ bản, thì model nhỏ là lựa chọn phù hợp hơn.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi **"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)? Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?** (3–4 câu)
> *Phản hồi 1 giải thích machine learning bằng hình ảnh ví von, giọng văn giàu cảm xúc, gần như không dùng thuật ngữ kỹ thuật và khá ngắn gọn. Phản hồi 2 giải thích theo định nghĩa chính xác, có thuật ngữ chuyên môn, ví dụ thực tế và có thể kèm đoạn code minh họa nên dài và chi tiết hơn. Hai phản hồi khác nhau rõ rệt về giọng văn, mức độ kỹ thuật và cách trình bày. Persona "nhà thơ" ưu tiên hình ảnh, cảm xúc và sự dễ hiểu, trong khi persona "kỹ sư senior" ưu tiên tính chính xác, cấu trúc logic và ví dụ kỹ thuật. Điều này cho thấy system prompt có thể điều khiển vai trò, phong cách diễn đạt, độ dài, mức độ chuyên môn và định dạng của phản hồi.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens` (tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa — và vì sao?**
> *Với tiếng Việt, số token thực tế đo bằng tiktoken thường cao hơn đáng kể (có thể gấp 2-3 lần) so với ước lượng thô (số từ / 0.75). Lý do là tiếng Việt có nhiều dấu câu, dấu thanh và là ngôn ngữ đơn âm tiết nhưng một từ có thể ghép từ nhiều âm tiết (ví dụ: "sinh viên"), nên một từ thường bị tách thành nhiều token (sub-words). Do đó, nếu dùng ước lượng thô để dự toán ngân sách API cho ứng dụng tiếng Việt, ta sẽ dự toán thiếu hụt lớn ngân sách so với thực tế.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi, (c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Ứng dụng hưởng lợi nhiều nhất từ streaming là (a) chatbot văn bản và (b) trợ lý giọng nói, vì streaming giúp phản hồi được hiển thị/đọc ngay lập tức từng phần, giảm cảm giác chờ đợi, mang lại trải nghiệm tương tác tự nhiên và mượt mà hơn. Ngược lại, ứng dụng (c) pipeline dịch tài liệu chạy ngầm ban đêm không cần streaming vì nó hoạt động ẩn, người dùng chỉ quan tâm đến kết quả cuối cùng khi hoàn tất, việc nhận dữ liệu từng phần không mang lại giá trị thêm.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu nhiên) giải quyết vấn đề gì còn sót lại?**
> *Khi API quá tải, exponential backoff giúp giảm áp lực tức thì lên server hiệu quả hơn nhiều so với delay cố định, vì nó trải dài các yêu cầu theo thời gian thay vì dồn dập gửi lại liên tục. Tuy nhiên, nếu nhiều client cùng thất bại một lúc, chúng có thể cùng retry vào cùng một thời điểm tương lai. Kỹ thuật "jitter" sẽ phân tán các lần retry này, tránh tình trạng tất cả client lại đồng loạt gửi request gây sập server lần nữa.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi đó:**
> *System prompt: "Bạn là một trợ lý ảo chuyên về ẩm thực Việt Nam, luôn nhiệt tình và sử dụng icon trong câu trả lời. Hãy đưa ra công thức nấu ăn chi tiết theo từng bước.". Nếu xóa phần "chuyên về ẩm thực Việt Nam": Trợ lý sẽ mất đi sự tập trung chuyên môn, có thể trả lời chung chung hoặc đưa ra các món ăn nước ngoài khi không được hỏi cụ thể. Nếu xóa phần "sử dụng icon trong câu trả lời": Giọng văn của trợ lý sẽ trở nên khô khan, thiếu sự thân thiện và sinh động vốn có của một trợ lý nhiệt tình.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Tình huống: Ở lượt 1, người dùng nói "Tên tôi là An, tôi đang ở Hà Nội". Ở lượt 2-5, họ hỏi về du lịch, thời tiết, ẩm thực. Đến lượt 6, họ hỏi "Gợi ý nhà hàng gần nhà tôi". Do chỉ giữ 4 lượt cuối, trợ lý không còn nhớ "Hà Nội" ở lượt 1 nên không thể trả lời chính xác. Cách khắc phục: Có thể dùng cơ chế tóm tắt định kỳ các thông tin quan trọng từ các lượt cũ (tên, địa điểm...) để đưa vào context, hoặc sử dụng vector database để lưu trữ và truy xuất các thông tin cá nhân/ngữ cảnh khi cần thiết.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
