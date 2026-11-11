# Frontend VisualizeLLM

## Chạy local

Từ thư mục `frontend`:

```bash
npm install
npm run dev
```

FastAPI phải chạy ở `127.0.0.1:8000`. Vite proxy chuyển `/api` tới backend trong development. Kiểm tra TypeScript và build bằng `npm run build`.

## Luồng dữ liệu

1. `GET /api/models` cung cấp bộ chọn model. `GET /api/model-info?model_id=...` tạo các lựa chọn layer/head từ config model đã chọn.
2. Nút **Chỉ tokenize** gọi `POST /api/tokenize`; không chạy model. Cả hai POST gửi `{ "text": "...", "model_id": "..." }`; bỏ model_id thì backend dùng Qwen2.5-0.5B.
3. Nút **Chạy model, tạo trace** gọi `POST /api/traces`. Response cung cấp trace ID, tokenization và model metadata.
4. Chọn layer gọi `GET /api/traces/{trace_id}/layers/{layer}/qkv` nếu layer chưa có trong cache. Cache được gắn với trace ID và xóa khi tạo input/trace mới.
5. Đổi head hoặc token chỉ chọn lát cắt dữ liệu trong frontend. Response đến muộn của một layer/trace cũ không được thay dữ liệu hiện tại.

Không có dữ liệu activation giả hoặc fallback: heatmap chỉ xuất hiện sau khi API trả trace QKV.

## Tensor và cách vẽ

- Contract `types/trace.ts` khớp schema FastAPI. Tên tensor là `q_attention_input`, `k_attention_input`, `v_attention_input`; stage là `attention_input`.
- Điểm lấy là đầu vào attention backend, sau những xử lý của implementation model. Với Qwen2 đây là Q/K sau RoPE. Model không dùng RoPE không được app tự thêm RoPE; nhãn chung không khẳng định mọi model đều có RoPE.
- Trục của `values`: `[token][head][head_dimension]`, batch size 1 đã được bỏ tại backend.
- `visualization/qkv.ts` kiểm tra shape/trục/stage và chọn query head cùng KV head theo GQA.
- Chiều Q/K lấy từ metadata config đã kiểm tra (ưu tiên `head_dim` explicit); V dùng chiều thực tế của tensor. Không giả định chiều head luôn bằng hidden size chia số query head.
- Plotly nhận ma trận `[token][head_dimension]`. Hàng 0 ở trên; nhãn hàng dùng position và raw token.
- Mặc định **Trừ trung bình theo token**: `X′[i,h,d] = X[i,h,d] - mean_tokens(X[:,h,d])`. Thống kê riêng từng tensor, layer, head và chiều vector; không gộp head hoặc chiều. Giữ đơn vị activation; dấu biểu thị cao/thấp hơn trung bình.
- **Z-score theo token**: `(X-mean)/std`, dùng population std (`correction=0`). Chiều có std bằng 0 được đặt 0. Thể hiện số độ lệch chuẩn, không thể đọc như độ lớn activation gốc; ít token có thể khuếch đại biến thiên nhỏ.
- **Min–max theo token**: `(X-min)/(max-min)` trên từng cặp `(head, dimension)` qua mọi token. Chiều hằng đặt 0. Dùng thang [0,1] cố định và bảng màu tuần tự: trắng là thấp, xanh đậm là cao; không biểu diễn dấu activation.
- Vẫn có **Δ so với token tham chiếu**: `ΔX[i,h,d] = X[i,h,d] - X[reference,h,d]`, và **Activation gốc**. Chọn token so sánh không ảnh hưởng thống kê mean/std/minmax. Hover và bảng luôn cho xem giá trị gốc lẫn giá trị biểu diễn; không sửa trace.
- Token tham chiếu do người dùng chọn, mặc định position 0; bấm hàng hoặc đổi head không tự đổi tham chiếu. Input một token thì Δ bằng 0.
- Mặc định dùng thang riêng từng tensor: max trị tuyệt đối của dữ liệu đang vẽ trên mọi token/head của tensor đó. Không so sánh độ lớn giữa Q/K/V bằng màu khi dùng thang riêng.
- Có tùy chọn thang chung Q/K/V. Ở chế độ gốc, dùng `color_limit` backend; ở chế độ Δ, dùng max trị tuyệt đối của mọi Δ trong cả ba tensor. Không chia chuẩn hóa giá trị.
- Thang cố định khi đổi head/token đang chọn. Đổi layer, chế độ hoặc token tham chiếu sẽ tính lại thang và giao diện ghi rõ quy tắc đó.
- So sánh hai token bằng số chiều khác nhau, max trị tuyệt đối của hiệu và bảng giá trị từng chiều. Hai hàng cùng màu không chứng minh hai vector bằng nhau.
- Hover dùng chuỗi giá trị từ JSON; bấm ô hiện số và chọn token. Viền hàng biểu thị lựa chọn, không thay đổi giá trị.
- Tensor có `source_dtype=float16` vẫn giữ độ chính xác FP16 dù backend chuyển sang FP32 để serialize JSON.

Chọn ô hoặc token không thể hiện quan hệ nhân quả hay ý nghĩa ngữ nghĩa của một chiều activation.

## Phạm vi tracer backend

Backend dùng `AttentionInterface` và formatter mask SDPA, không monkey-patch hàm RoPE riêng của model. Hỗ trợ phạm vi decoder-only MHA/GQA truyền kwargs và gọi attention đúng một lần cho mỗi decoder layer; trường hợp ngoài phạm vi phải báo lỗi rõ ràng.

Bộ chọn lấy từ `backend/model_catalog.py`: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m), [GPT-2](https://huggingface.co/openai-community/gpt2). Test kiến trúc dùng cấu hình nhỏ ngẫu nhiên; không đồng nghĩa đã chạy mọi pretrained checkpoint trên GPU. Lần đầu cần kết nối Hugging Face để tải tokenizer/config và trọng số.

Đổi model xóa kết quả/cache frontend. Backend giữ tối đa một model GPU, giải phóng model trước khi nạp model khác dưới `trace_lock`. Chỉ chọn model sẽ đọc config; trọng số chỉ nạp khi tạo trace. Trace CPU trước vẫn tồn tại tới khi một trace mới được tạo thành công.

Sau khi đổi contract từ pre-RoPE sang attention-input, restart backend và tạo trace mới. Frontend không diễn giải trace cũ bằng nhãn mới.
