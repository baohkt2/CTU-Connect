# Kiến trúc hệ thống CTU-Connect

Dưới đây là sơ đồ kiến trúc tổng quát của hệ thống CTU-Connect và mô tả chi tiết từng thành phần.

```mermaid
flowchart LR
  Browser[Người dùng (Browser / Mobile App)] -->|HTTPS| CDN[CDN / Nginx / Reverse Proxy]
  CDN --> API_GW[API Gateway / Backend for Frontend]
  API_GW --> Auth[`Auth Service (OAuth / JWT)`]
  API_GW --> User[`User Service (REST)`]
  API_GW --> Friend[`Friend Service`]
  API_GW --> Chat[`Chat Service (Realtime)`]
  API_GW --> Recommend[`Recommend Service`]
  API_GW --> Notification[`Notification Service`]
  Chat -->|WebSocket| Browser

  subgraph Infra
    Postgres[(Postgres DB)]
    Redis[(Redis Cache / Session)]
    MinIO[(Object Storage / S3)]
    Kafka[(Kafka / Message Bus)]
  end

  User --> Postgres
  Friend --> Postgres
  Auth --> Postgres
  Chat --> Redis
  Chat --> Postgres
  Chat --> Kafka
  Recommend --> Kafka
  Recommend --> Redis
  Recommend --> Postgres
  Recommend --> MinIO
  Notification --> Kafka
  Notification --> MinIO

  subgraph RecommendService
    RJ[Java API (`recommend-service/java-api`)]
    RM[Python Model (`recommend-service/python-model`)]
    RJ --> RM
    RM --> Redis
    RM --> Postgres
  end

  Kafka -->|events| Notification
  Kafka -->|events| Recommend

  CI[CI/CD (GitHub Actions / Jenkins)] -.-> API_GW
  CI -.-> Recommend
  CI -.-> Chat

  style CDN fill:#f3f4f6,stroke:#ccc
  style API_GW fill:#e6f7ff,stroke:#7fbfff
  style RecommendService fill:#fffbe6,stroke:#ffd86b
``` 

**Tổng quan luồng:**
- Người dùng truy cập qua `Browser` hoặc `Mobile App` -> yêu cầu qua `CDN/Nginx` -> vào `API Gateway`.
- `API Gateway` phân phối request tới các microservice: `Auth`, `User`, `Friend`, `Chat`, `Recommend`, `Notification`.
- `Chat Service` dùng WebSocket cho realtime messaging, lưu nhanh ở `Redis` và bền ở `Postgres`. Thông điệp quan trọng được phát qua `Kafka`.
- `Recommend Service` gồm `java-api` (giao diện REST) và `python-model` (inference). Dữ liệu embedding, cache và mô hình được lưu trên `Redis`/`MinIO`/`Postgres`.
- `Notification Service` tiêu thụ sự kiện từ `Kafka` để gửi push/email và lưu file media lên `MinIO`.
- CI/CD chịu trách nhiệm build và deploy các service (Docker images, K8s hoặc Docker Compose).

**Mô tả chi tiết các thành phần:**
- `CDN / Nginx / Reverse Proxy`:
  - Chịu SSL termination, routing, rate limiting, static caching.
  - Thường cấu hình trong [docker-compose.yml](docker-compose.yml) hoặc trên ingress controller khi dùng Kubernetes.

- `API Gateway / Backend for Frontend`:
  - Authentication check (validate `JWT`), request routing, response aggregation, throttling.
  - Có thể là một lightweight proxy (NGINX + Lua), Kong, Traefik hoặc custom Node/Java layer.

- `Auth Service`:
  - Quản lý đăng nhập, đăng ký, refresh token, OAuth integration (SSO).
  - Lưu thông tin user auth trong `Postgres` và sessions vào `Redis` (nếu dùng session-based).

- `User Service`:
  - CRUD thông tin người dùng, profile, privacy settings. Lưu chính trong `Postgres`.

- `Friend Service`:
  - Quản lý friend request, following, mutuals. Ghi vào `Postgres`, cache queries phổ biến vào `Redis`.

- `Chat Service`:
  - Hỗ trợ WebSocket (Socket.IO, WS hoặc gRPC streaming) cho realtime messages.
  - Thiết kế: short-term messages vào `Redis` để truy vấn nhanh; persist vào `Postgres` cho lịch sử.
  - Kiến trúc event-driven: gửi sự kiện message vào `Kafka` để các consumer (search index, notifications, analytics) xử lý.

- `Recommend Service`:
  - Gồm hai thành phần chính: `java-api` (giao diện REST) và `python-model` (inference / embedding).
  - `python-model` thực thi inference, truy cập model artifacts trong `recommend-service/model` hoặc `MinIO`.
  - Cache kết quả gợi ý trong `Redis`. Dữ liệu huấn luyện / bảng embedding lưu trong `Postgres` hoặc object storage.
  - Tham khảo chi tiết trong [recommend-service/ARCHITECTURE.md](recommend-service/ARCHITECTURE.md).

- `Notification Service`:
  - Gửi email, push, in-app notifications. Tiêu thụ events từ `Kafka`.
  - Lưu media attachments vào `MinIO`.

- `Kafka` (Message Bus):
  - Dùng cho event-driven workflows: message events, user-activity, recommendation triggers, analytics.
  - Tách rời producers/consumers giúp mở rộng độc lập.

- `Redis`:
  - Cache, session store, rate-limit counters, fast lookup cho realtime chat/recommend.

- `Postgres`:
  - Cơ sở dữ liệu chính cho user, friends, chat history, metadata.

- `MinIO` / S3:
  - Lưu trữ tệp, ảnh, model artifacts.

- `CI/CD`:
  - Build images (`recommend-java.Dockerfile`, `recommend-python.Dockerfile`), push registry, deploy (Docker Compose cho dev, K8s cho prod).
  - Tham khảo `recommend-service/docker/`.

**Triển khai & môi trường phát triển:**
- Dev: Docker Compose trong `recommend-service/docker` và root `docker-compose.yml` để chạy nhanh toàn hệ thống.
- Prod: K8s (Helm charts) hoặc ECS với autoscaling cho `Chat`, `Recommend`, `API Gateway`.
- Tài nguyên quan trọng: scaling `Chat` pods với sticky sessions hoặc external websocket gateway; `Kafka` cluster sizing; Redis cluster cho HA.

**Bảo mật & Observability:**
- Bảo mật: HTTPS everywhere, JWT/OAuth, RBAC cho service-to-service, secrets từ Vault/KeyVault.
- Observability: centralized logging (ELK/EFK), tracing (Jaeger), metrics (Prometheus + Grafana).

**Gợi ý cải tiến/ghi chú kỹ thuật:**
- Tách read/write DB (CQRS) cho chat nếu cần throughput lớn.
- Dùng stream processing (Kafka Streams / Flink) cho real-time metrics và offline feature updates cho recommender.
- Đóng gói model inference trong k8s as a service hoặc tận dụng GPU nodes nếu cần inference nặng.

---

Tệp này là bản phác thảo sơ đồ + mô tả chi tiết. Nếu bạn muốn, tôi có thể:
- Xuất sơ đồ Mermaid ra PNG/SVG.
- Mở rộng chi tiết từng service (`Chat`, `Recommend`) thành sơ đồ sequence hoặc component riêng.
- Thêm lược đồ deploy (K8s manifests / Helm).

File liên quan trong workspace:
- [recommend-service/ARCHITECTURE.md](recommend-service/ARCHITECTURE.md)
- [recommend-service/docker/docker-compose.yml](recommend-service/docker/docker-compose.yml)
- [docker-compose.yml](docker-compose.yml)


