-- Test Data for Recommendation Service
-- Load this into PostgreSQL: docker exec -i postgres-recommend-dev psql -U postgres -d recommendation_db < test-data.sql

-- Create users metadata table (graph relationships are in Neo4j)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    faculty_id VARCHAR(50),
    major_id VARCHAR(50),
    batch_id VARCHAR(20)
);

-- Insert test users
INSERT INTO users (id, username, faculty_id, major_id, batch_id) VALUES
('11111111-1111-1111-1111-111111111111', 'nguyen_van_a', 'CNTT', 'KTPM', '2021'),
('22222222-2222-2222-2222-222222222222', 'tran_thi_b', 'CNTT', 'KTPM', '2021'),
('33333333-3333-3333-3333-333333333333', 'le_van_c', 'CNTT', 'HTTT', '2021'),
('44444444-4444-4444-4444-444444444444', 'pham_thi_d', 'CNTT', 'KHMT', '2022'),
('55555555-5555-5555-5555-555555555555', 'hoang_van_e', 'CNTT', 'KTPM', '2021')
ON CONFLICT (id) DO NOTHING;

-- Insert test posts with embeddings (vector will be computed by service)
INSERT INTO post_embeddings (id, post_id, author_id, content, academic_score, popularity_score, created_at, updated_at)
VALUES
-- High academic content
(gen_random_uuid(), 'post-001', '11111111-1111-1111-1111-111111111111', 
 'Nghiên cứu về thuật toán Machine Learning trong xử lý ngôn ngữ tự nhiên. Bài viết trình bày các phương pháp deep learning hiện đại như Transformer, BERT, GPT và ứng dụng trong phân tích văn bản tiếng Việt.', 
 0.95, 0.8, NOW() - INTERVAL '2 days', NOW()),

(gen_random_uuid(), 'post-002', '22222222-2222-2222-2222-222222222222',
 'Hướng dẫn chi tiết sử dụng Spring Boot và PostgreSQL để xây dựng RESTful API. Bao gồm configuration, JPA, Hibernate, và best practices trong phát triển backend Java.',
 0.90, 0.7, NOW() - INTERVAL '1 day', NOW()),

(gen_random_uuid(), 'post-003', '33333333-3333-3333-3333-333333333333',
 'Cuộc thi lập trình CTU Code War 2024 - Giải thưởng hấp dẫn lên đến 20 triệu đồng. Đăng ký ngay để thể hiện kỹ năng coding và networking với các cao thủ.',
 0.85, 0.9, NOW() - INTERVAL '3 hours', NOW()),

(gen_random_uuid(), 'post-004', '11111111-1111-1111-1111-111111111111',
 'Học bổng toàn phần du học Nhật Bản MEXT 2025 dành cho sinh viên CNTT. Hỗ trợ 100% học phí, sinh hoạt phí và vé máy bay. Deadline: 31/12/2024.',
 0.92, 0.75, NOW() - INTERVAL '5 hours', NOW()),

(gen_random_uuid(), 'post-005', '44444444-4444-4444-4444-444444444444',
 'Tips và tricks ôn thi cuối kỳ môn Cấu trúc dữ liệu và Giải thuật. Tổng hợp các dạng bài tập hay gặp về Tree, Graph, Dynamic Programming.',
 0.88, 0.65, NOW() - INTERVAL '1 day', NOW()),

(gen_random_uuid(), 'post-006', '22222222-2222-2222-2222-222222222222',
 'Sự kiện giao lưu sinh viên IT toàn quốc tại CTU - IT Connect 2024. Workshop, hackathon, và cơ hội kết nối với doanh nghiệp.',
 0.70, 0.95, NOW() - INTERVAL '6 hours', NOW()),

(gen_random_uuid(), 'post-007', '33333333-3333-3333-3333-333333333333',
 'Review chi tiết khóa học Python for Data Science trên Coursera. Nội dung hay, bài tập thực hành nhiều. Sau khóa học có thể làm được data analysis project.',
 0.87, 0.60, NOW() - INTERVAL '2 days', NOW()),

(gen_random_uuid(), 'post-008', '11111111-1111-1111-1111-111111111111',
 'Bài báo khoa học "Deep Learning for Vietnamese Text Classification" được công bố trên IEEE Transactions. Citation impact factor 8.5.',
 0.98, 0.50, NOW() - INTERVAL '1 week', NOW()),

-- Medium academic
(gen_random_uuid(), 'post-009', '55555555-5555-5555-5555-555555555555',
 'Kinh nghiệm phỏng vấn intern Backend Developer tại VNG, FPT, Viettel. Những câu hỏi thường gặp và cách chuẩn bị.',
 0.75, 0.85, NOW() - INTERVAL '1 day', NOW()),

(gen_random_uuid(), 'post-010', '22222222-2222-2222-2222-222222222222',
 'Top 10 công cụ dev tools hữu ích cho sinh viên CNTT: Git, Docker, Postman, VSCode extensions...',
 0.72, 0.78, NOW() - INTERVAL '8 hours', NOW()),

-- Low academic (social/entertainment)
(gen_random_uuid(), 'post-011', '44444444-4444-4444-4444-444444444444',
 'Quán cafe view đẹp gần trường CTU, phù hợp cho nhóm học tập và làm bài tập nhóm.',
 0.30, 0.88, NOW() - INTERVAL '4 hours', NOW()),

(gen_random_uuid(), 'post-012', '33333333-3333-3333-3333-333333333333',
 'Tìm bạn cùng đi gym gần ký túc xá CTU. Buổi tối từ 6-8h.',
 0.20, 0.65, NOW() - INTERVAL '12 hours', NOW())

ON CONFLICT (post_id) DO NOTHING;

-- Display inserted data
SELECT 
    post_id, 
    LEFT(content, 60) as content_preview, 
    academic_score, 
    popularity_score,
    created_at 
FROM post_embeddings 
ORDER BY created_at DESC;

-- Show summary
SELECT 
    COUNT(*) as total_posts,
    AVG(academic_score) as avg_academic_score,
    AVG(popularity_score) as avg_popularity_score
FROM post_embeddings;
