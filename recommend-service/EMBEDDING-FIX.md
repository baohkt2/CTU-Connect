# Embedding Vector Fix - Giải quyết vấn đề Vector bằng 0

## Vấn đề
Sau khi kiểm tra database `recommend_db` tại bảng `post_embeddings`, phát hiện tất cả các vector đều bằng 0 (zero vector).

## Nguyên nhân
1. **Service PhoBERT không chạy**: `EmbeddingService` được cấu hình gọi PhoBERT service tại `http://localhost:8096` nhưng service này không chạy
2. **Fallback embedding**: Khi kết nối thất bại, code tự động trả về zero vector (768 chiều đều là 0) thay vì báo lỗi
3. **Python service đang chạy nhưng không được sử dụng**: Python AI service đang chạy trên port 8000 với endpoint `/embed/post` nhưng Java service không gọi đến

## Giải pháp
Cập nhật `EmbeddingService.java` để sử dụng Python AI service thay vì PhoBERT service:

### Thay đổi chính

#### 1. Thay đổi cấu hình service URL
**Trước:**
```java
@Value("${recommendation.nlp.phobert-service-url}")
private String phoBertServiceUrl;  // http://localhost:8096

@Value("${recommendation.nlp.embedding-endpoint}")
private String embeddingEndpoint;  // /api/nlp/embed
```

**Sau:**
```java
@Value("${recommendation.python-service.url:http://localhost:8000}")
private String pythonServiceUrl;  // http://localhost:8000

// Endpoint cố định: /embed/post
```

#### 2. Thay đổi request format
**Trước:**
```java
EmbeddingRequest request = EmbeddingRequest.builder()
    .text(text)
    .model("phobert")
    .normalize(true)
    .build();
```

**Sau:**
```java
Map<String, Object> request = new HashMap<>();
request.put("post_id", postId != null ? postId : "");
request.put("content", text);
request.put("title", "");
```

#### 3. Xử lý response từ Python service
```java
Map<String, Object> response = webClient.post()
    .uri("/embed/post")
    .bodyValue(request)
    .retrieve()
    .bodyToMono(Map.class)
    .timeout(Duration.ofMillis(timeout))
    .block();

if (response != null && response.containsKey("embedding")) {
    Object embeddingObj = response.get("embedding");
    // Parse List<Number> to float[]
    if (embeddingObj instanceof java.util.List) {
        java.util.List<?> list = (java.util.List<?>) embeddingObj;
        embedding = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            embedding[i] = ((Number) list.get(i)).floatValue();
        }
    }
}
```

#### 4. Xóa fallback embedding
- Xóa method `createFallbackEmbedding()` 
- Thay vào đó throw exception khi có lỗi
- Điều này đảm bảo không lưu zero vector vào database

### Files thay đổi
- `EmbeddingService.java` - Cập nhật logic gọi Python service

### Cấu hình đã có sẵn
File `application-dev.yml` đã có cấu hình Python service:
```yaml
recommendation:
  python-service:
    url: http://localhost:8000
    predict-endpoint: /api/model/predict
    timeout: 10000
    enabled: true
    fallback-to-legacy: true
```

## Kiểm tra

### 1. Kiểm tra Python service
```bash
curl -X POST http://localhost:8000/embed/post \
  -H "Content-Type: application/json" \
  -d '{"post_id":"test","content":"Công nghệ thông tin","title":""}'
```

**Kết quả mong đợi:**
```json
{
  "id": "test",
  "embedding": [-0.241, 1.455, 2.606, ...], // 768 số thực
  "dimension": 768
}
```

### 2. Kiểm tra sau khi restart service
1. Restart recommendation service Java
2. Tạo một post mới trong post-service
3. Kiểm tra database `recommend_db`:
```sql
SELECT post_id, 
       length(embedding_vector) as vector_length,
       substring(embedding_vector, 1, 100) as vector_preview
FROM post_embeddings
ORDER BY created_at DESC
LIMIT 1;
```

**Kết quả mong đợi:**
- `vector_length` > 100 ký tự (không phải "[0,0,0,...]")
- `vector_preview` chứa các số thực khác 0

### 3. Kiểm tra log
```bash
tail -f logs/recommendation-service-dev.log | grep -i "embedding\|python"
```

**Log mong đợi:**
```
Calling Python service at: http://localhost:8000/embed/post
Successfully generated embedding for post: xxx (dimension: 768)
```

## Lợi ích
1. **Vector thực**: Embedding được tạo bởi PhoBERT model thật sự, không còn zero vector
2. **Tích hợp tốt hơn**: Sử dụng Python AI service đã có sẵn thay vì cần service PhoBERT riêng
3. **Xử lý lỗi rõ ràng**: Throw exception khi không tạo được embedding thay vì âm thầm lưu zero vector
4. **Logging tốt hơn**: Log rõ ràng khi gọi Python service và kết quả

## Lưu ý
- **Python service phải chạy**: Đảm bảo Python service đang chạy trên port 8000
- **Xóa dữ liệu cũ**: Các bản ghi với zero vector cũ nên được xóa hoặc cập nhật:
```sql
DELETE FROM post_embeddings 
WHERE embedding_vector LIKE '[0,0,0%';
```
- **Kafka events**: Các post mới sẽ tự động có embedding đúng qua Kafka consumer

## Build Status
✅ Build thành công
✅ Python service test thành công
✅ Embedding vector 768 chiều với giá trị thực

## Date
December 9, 2025
