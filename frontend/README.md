# Frontend VisualizeLLM

## Chạy local

Từ thư mục `frontend`:

```bash
npm install
npm run dev
```

FastAPI phải chạy ở `127.0.0.1:8000`. Vite proxy chuyển `/api` tới backend trong development. Kiểm tra TypeScript và build bằng `npm run build`.

## Luồng dữ liệu

1. `GET /api/model-info` tạo các lựa chọn layer/head từ config.
2. Nút **Chỉ tokenize** gọi `POST /api/tokenize`; không chạy model.
3. Nút **Chạy model, tạo trace** gọi `POST /api/traces`. Response cung cấp trace ID, tokenization và model metadata.
4. Chọn layer gọi `GET /api/traces/{trace_id}/layers/{layer}/qkv` nếu layer chưa có trong cache. Cache được gắn với trace ID và xóa khi tạo input/trace mới.
5. Đổi head hoặc token chỉ chọn lát cắt dữ liệu trong frontend. Response đến muộn của một layer/trace cũ không được thay dữ liệu hiện tại.

Không có dữ liệu activation giả hoặc fallback: heatmap chỉ xuất hiện sau khi API trả trace QKV.

## Tensor và cách vẽ

- Contract `types/trace.ts` khớp schema FastAPI. Q/K trước RoPE; V sau projection và không áp dụng RoPE.
- Trục của `values`: `[token][head][head_dimension]`, batch size 1 đã được bỏ tại backend.
- `visualization/qkv.ts` kiểm tra shape/trục/stage và chọn query head cùng KV head theo GQA.
- Plotly nhận ma trận `[token][head_dimension]`. Hàng 0 ở trên; nhãn hàng dùng position và raw token.
- Mặc định hiển thị **Δ so với token tham chiếu**: `ΔX[i,h,d] = X[i,h,d] - X[reference,h,d]`. Đây là dữ liệu dẫn xuất để thấy khác biệt giữa token, không phải activation gốc hay bước tính của model. Có thể chuyển về **Activation gốc**. Hover và bảng luôn cho xem giá trị gốc; không sửa trace.
- Token tham chiếu do người dùng chọn, mặc định position 0; bấm hàng hoặc đổi head không tự đổi tham chiếu. Input một token thì Δ bằng 0.
- Mặc định dùng thang riêng từng tensor: max trị tuyệt đối của dữ liệu đang vẽ trên mọi token/head của tensor đó. Không so sánh độ lớn giữa Q/K/V bằng màu khi dùng thang riêng.
- Có tùy chọn thang chung Q/K/V. Ở chế độ gốc, dùng `color_limit` backend; ở chế độ Δ, dùng max trị tuyệt đối của mọi Δ trong cả ba tensor. Không chia chuẩn hóa giá trị.
- Thang cố định khi đổi head/token đang chọn. Đổi layer, chế độ hoặc token tham chiếu sẽ tính lại thang và giao diện ghi rõ quy tắc đó.
- So sánh hai token bằng số chiều khác nhau, max trị tuyệt đối của hiệu và bảng giá trị từng chiều. Hai hàng cùng màu không chứng minh hai vector bằng nhau.
- Hover dùng chuỗi giá trị từ JSON; bấm ô hiện số và chọn token. Viền hàng biểu thị lựa chọn, không thay đổi giá trị.
- Tensor có `source_dtype=float16` vẫn giữ độ chính xác FP16 dù backend chuyển sang FP32 để serialize JSON.

Chọn ô hoặc token không thể hiện quan hệ nhân quả hay ý nghĩa ngữ nghĩa của một chiều activation.
