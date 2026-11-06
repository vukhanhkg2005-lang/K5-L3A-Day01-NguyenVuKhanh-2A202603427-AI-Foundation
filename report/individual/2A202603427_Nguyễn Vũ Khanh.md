# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.975   0.794      0.875        0.974   
2. BoW + MLP (deep learning)         0.956   0.897      0.926        0.947   
3a. Encoder (MiniLM) + kNN           0.881   0.918      0.899        0.952   
3b. Encoder (MiniLM) + LogReg        0.929   0.938      0.933        1.000   
3c. Encoder (MiniLM) + MLP head      0.929   0.938      0.933        1.000   
4a. LLM nhỏ zero-shot                0.909   1.000      0.952        0.952   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            9.192       5867.860       0.048   
2. BoW + MLP (deep learning)        16.072      32378.829       0.299   
3a. Encoder (MiniLM) + kNN         107.315         19.671      44.063   
3b. Encoder (MiniLM) + LogReg      107.735         30.748      24.463   
3c. Encoder (MiniLM) + MLP head    108.251         30.820      24.018   
4a. LLM nhỏ zero-shot                0.000          1.310     650.498   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.084         0.000  
2. BoW + MLP (deep learning)          0.453         0.000  
3a. Encoder (MiniLM) + kNN           58.852         0.000  
3b. Encoder (MiniLM) + LogReg        32.355         0.000  
3c. Encoder (MiniLM) + MLP head      33.765         0.000  
4a. LLM nhỏ zero-shot              1270.636     16173.333  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
Dự đoán 1 – MLP: Em nghĩ Validation Loss sẽ giảm mãi và MLP kém Logistic Regression. Thực tế, Validation Loss thấp nhất ở epoch 14 và MLP đạt F1 = 0.926, cao hơn Logistic Regression là 0.875. Em nhận ra mô hình có thể bị quá khớp nếu huấn luyện quá lâu.

Dự đoán 2 – Embedding: Em nghĩ Embedding + LogReg sẽ dự đoán đúng nhiều hơn BoW. Thực tế, cả hai đều đúng 5/6 trường hợp. Em hiểu rằng Embedding không phải lúc nào cũng tốt hơn BoW.

Dự đoán 3 – Stress test: Em nghĩ LLM sẽ khó xử lý Prompt Injection. Thực tế, LLM zero-shot đạt F1 = 0.983 và LLM phòng thủ đạt 1.000 trên 50 case khó. Em bất ngờ vì LLM có thể phân biệt chỉ dẫn giả với nội dung tin nhắn.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
BoW + LogReg tụt F1 nhiều nhất, từ 0.974 xuống 0.378. Nhóm khó nhất đối với hai mô hình BoW khi xét chung là spam giọng tự nhiên, với độ chính xác trung bình chỉ 15%.

Ví dụ thứ nhất: Tin nhắn giả làm bạn cũ để quảng cáo sản phẩm chăm sóc da. BoW dễ nhận nhầm vì tin nhắn có cách viết thân thiện và ít từ khóa spam.

Ví dụ thứ hai: Tin nhắn viết C0ngr4tulat10ns thay cho Congratulations. BoW khó nhận diện vì các ký tự đã bị thay đổi nên không khớp với từ khóa đã học.


## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Theo SEED = 9601, em nhận được đề số 2
Cách làm: Bước 1: Tạo Vocabulary với ba kích thước khác nhau bằng hàm build_vocab(texts_tr, max_size=size).
Bước 2: Chuyển dữ liệu train và test thành vector bằng hàm vectorize().
Bước 3: Huấn luyện mô hình Logistic Regression với 300 epochs và learning rate = 0.5.
Bước 4: Dự đoán nhãn trên tập Test với ngưỡng 0.5, sau đó tính Accuracy, Precision, Recall và F1-score.
Kết quả: 
Vocabulary 100: F1 = 0.805
Vocabulary 2000: F1 = 0.875
Vocabulary 5000: F1 = 0.875
Giải thích: Vocabulary 100 quá nhỏ nên bỏ sót nhiều từ quan trọng. Khi tăng lên 2.000, F1 tăng từ 0.805 lên 0.875. Tăng tiếp lên 5.000 không cải thiện kết quả. Vì vậy, tôi chọn Vocabulary 2.000 để tiết kiệm tài nguyên.
## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Em chọn Embedding + Logistic Regression vì mô hình có F1 SUB = 1.000, F1 test = 0.933 và p95 = 32.36 ms, đáp ứng yêu cầu dưới 50 ms.
Tuy nhiên, tốc độ xử lý chỉ khoảng 31 tin/giây, thấp hơn nhu cầu trung bình 116 tin/giây. Vì vậy cần tối thiểu 4 máy xử lý tương đương và có thêm dự phòng công suất.
BoW + LogReg nhanh hơn nhiều nhưng F1 trên 50 case khó chỉ đạt 0.378. LLM có chất lượng cao nhưng p95 lên tới 1.270,64 ms nên không đáp ứng yêu cầu độ trễ.
Em chấp nhận sử dụng nhiều tài nguyên hơn để đổi lấy chất lượng phân loại tốt hơn.
## 7. AI log và điều đã học
| Phần 4.1 – Kết nối API | Hướng dẫn cách sử dụng Gemini API trong Google Colab, lưu API Key vào Secrets và cấu hình LLM_PROVIDER. | Hướng dẫn thêm GEMINI_API_KEY vào Colab Secrets, bật Notebook access và đổi LLM_PROVIDER thành "gemini". | Kiểm tra lại tên biến API Key, cấu hình model và chạy thử kết nối API. Không hiển thị API Key trong notebook. |
| Phần 4.2 – Prompt phân loại | Hướng dẫn tạo prompt để LLM phân loại tin nhắn thành SPAM hoặc HAM bằng phương pháp zero-shot và few-shot. | Đề xuất prompt mô tả nhiệm vụ, yêu cầu AI chỉ trả lời một từ spam hoặc ham. Few-shot sử dụng thêm 4 ví dụ mẫu để AI hiểu cách phân loại. | Kiểm tra prompt có đủ nội dung, đúng định dạng và kết quả trả về có thể đọc được bằng hàm parse_label. |
| Phần 4.4 – Đánh giá và chi phí | Hướng dẫn cách chạy LLM trên tập SUB, tính F1-score, thời gian phản hồi và chi phí sử dụng API. | Giải thích cách gọi API cho từng tin nhắn, lấy kết quả dự đoán, đo độ trễ và ước tính chi phí dựa trên số token đầu vào, đầu ra. | Đối chiếu kết quả chạy thực tế, kiểm tra nhãn dự đoán, F1-score, thời gian phản hồi và chi phí API. |
*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
Điều em học được:Em hiểu rằng mỗi mô hình AI có cách xử lý văn bản khác nhau. BoW dựa vào từ khóa, Embedding dựa vào ý nghĩa và LLM có khả năng hiểu ngữ cảnh tốt hơn. Em cũng hiểu rằng không nên chỉ dựa vào Accuracy mà cần đánh giá thêm F1, độ trễ và chi phí trước khi chọn mô hình cho sản phẩm thực tế.
