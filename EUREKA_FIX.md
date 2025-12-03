# Fix Eureka Communication Between Docker and Local Services

## Vấn đề
- **api-gateway** và **eureka-server** chạy trên Docker
- **auth-service**, **user-service**, **media-service**, **post-service** chạy trên local
- Lỗi: `No servers available for service: auth-service`

## Nguyên nhân
Khi các service local đăng ký với Eureka, chúng sử dụng hostname `localhost`. Khi api-gateway (chạy trong Docker) cố gắng gọi các service này qua Eureka discovery, nó nhận được địa chỉ `localhost` - nhưng `localhost` trong Docker container trỏ đến chính container đó, không phải máy host.

## Giải pháp

### 1. Cấu hình Docker Compose
Thêm `extra_hosts` vào api-gateway để map `host.docker.internal` tới IP của máy host:

```yaml
api-gateway:
  extra_hosts:
    - "host.docker.internal:host-gateway"
```

### 2. Cấu hình Local Services
Thay đổi Eureka instance configuration trong tất cả các service chạy local:

**Trước:**
```properties
eureka.instance.prefer-ip-address=true
```

**Sau:**
```properties
eureka.instance.prefer-ip-address=false
eureka.instance.hostname=host.docker.internal
```

### 3. Files đã thay đổi
- `docker-compose.yml`: Thêm extra_hosts cho api-gateway
- `auth-service/src/main/resources/application.properties`
- `user-service/src/main/resources/application.properties`
- `media-service/src/main/resources/application.properties`
- `post-service/src/main/resources/application.properties`

## Cách hoạt động

1. **Services local** đăng ký với Eureka sử dụng hostname `host.docker.internal`
2. **Eureka server** (trong Docker) lưu thông tin service với hostname `host.docker.internal`
3. **API Gateway** (trong Docker) query Eureka và nhận được địa chỉ `host.docker.internal`
4. Nhờ `extra_hosts`, Docker resolve `host.docker.internal` thành IP của máy host
5. API Gateway có thể kết nối thành công với services trên local

## Cách test

1. **Build lại api-gateway** (vì có thay đổi docker-compose):
```bash
docker-compose up -d --build api-gateway
```

2. **QUAN TRỌNG - Restart các services local** để áp dụng cấu hình mới:
```bash
# BẮT BUỘC phải restart tất cả services local để load config mới
# Nếu không restart, services vẫn sẽ dùng hostname máy cũ thay vì host.docker.internal
# Restart từng service hoặc sử dụng IDE để restart
```

3. **Kiểm tra Eureka Dashboard**:
- Truy cập: http://localhost:8761
- Xác nhận các service đã đăng ký với hostname `host.docker.internal`

4. **Test API call**:
```bash
curl http://localhost:8090/api/auth/me
```

## Lưu ý quan trọng

1. **Port mapping**: Đảm bảo các service local chạy đúng port như trong cấu hình:
   - auth-service: 8080
   - user-service: 8081
   - media-service: 8084
   - post-service: 8085

2. **Eureka Server**: Phải đảm bảo eureka-server đã chạy và healthy trước khi start các service khác

3. **Kafka Configuration**: media-service đã được sửa để dùng `localhost:9092` thay vì `kafka:9092` vì chạy trên local

4. **Windows/Mac/Linux**: `host.docker.internal` hoạt động tốt trên Docker Desktop (Windows/Mac). Trên Linux có thể cần cấu hình khác hoặc dùng IP thực của máy host.
