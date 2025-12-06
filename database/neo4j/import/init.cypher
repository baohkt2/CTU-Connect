// -----------------------------------------------------------------
// 0. CLEAN SLATE (Optional: Use with caution to reset the database)
// -----------------------------------------------------------------
MATCH (n) DETACH DELETE n;

// =================================================================
// 1. CONSTRAINTS & INDEXES
// =================================================================
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (uni:University) REQUIRE uni.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:College) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (f:Faculty) REQUIRE f.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:Major) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (b:Batch) REQUIRE b.year IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gender) REQUIRE g.name IS UNIQUE;

// =================================================================
// 2. CREATE HIERARCHICAL UNIVERSITY STRUCTURE
// =================================================================
MERGE (uni:University {name: 'Đại học Cần Thơ', established: 1966})

WITH uni, [
{ college: 'Khoa Công nghệ Thông tin và Truyền thông', faculties: [
{ faculty: 'Bộ môn Công nghệ Phần mềm', majors: ['Công nghệ Phần mềm', 'Khoa học Máy tính'] },
{ faculty: 'Bộ môn Hệ thống Thông tin', majors: ['Hệ thống Thông tin', 'Kinh doanh Kỹ thuật Số'] },
{ faculty: 'Bộ môn Truyền thông Dữ liệu và Mạng máy tính', majors: ['Mạng máy tính và Truyền thông dữ liệu'] }
]},
{ college: 'Khoa Kinh tế', faculties: [
{ faculty: 'Bộ môn Kinh tế học', majors: ['Kinh tế học'] },
{ faculty: 'Bộ môn Quản trị Kinh doanh', majors: ['Marketing', 'Quản trị Kinh doanh'] },
{ faculty: 'Bộ môn Tài chính Ngân hàng', majors: ['Tài chính - Ngân hàng'] }
]},
{ college: 'Khoa Kỹ thuật', faculties: [
{ faculty: 'Bộ môn Kỹ thuật Xây dựng', majors: ['Kỹ thuật Xây dựng'] },
{ faculty: 'Bộ môn Kỹ thuật Cơ khí', majors: ['Kỹ thuật Cơ khí', 'Robot và Trí tuệ nhân tạo'] }
]},
{ college: 'Khoa Nông nghiệp', faculties: [] },
{ college: 'Khoa Y Dược', faculties: [] },
{ college: 'Khoa Sư phạm', faculties: [] },
{ college: 'Khoa Khoa học Tự nhiên', faculties: [] },
{ college: 'Khoa Khoa học Xã hội và Nhân văn', faculties: [] },
{ college: 'Khoa Luật', faculties: [] }
] AS universityStructure

UNWIND universityStructure AS collegeData
MERGE (c:College {name: collegeData.college})
MERGE (uni)-[:HAS_COLLEGE]->(c)

WITH collegeData, c
WHERE size(collegeData.faculties) > 0
UNWIND collegeData.faculties AS facultyData
MERGE (f:Faculty {name: facultyData.faculty, college: collegeData.college})
MERGE (c)-[:HAS_FACULTY]->(f)

WITH facultyData, f
UNWIND facultyData.majors AS majorName
MERGE (m:Major {name: majorName, faculty: facultyData.faculty})
MERGE (f)-[:HAS_MAJOR]->(m);

// =================================================================
// 3. CREATE BATCH (COURSE YEAR) NODES
// =================================================================
WITH [2021, 2022, 2023, 2024, 2025] AS batches
MATCH (uni:University {name: 'Đại học Cần Thơ'})
UNWIND batches AS batchYear
MERGE (b:Batch {year: toInteger(batchYear)})
MERGE (uni)-[:HAS_BATCH]->(b);

// =================================================================
// 4. CREATE GENDER NODES
// =================================================================
MERGE (g1:Gender {name: 'Nam'})
MERGE (g2:Gender {name: 'Nữ'});

// =================================================================
// 5. CREATE SAMPLE USERS (UUID FIXED) - WITH ALL REQUIRED FIELDS
// =================================================================
WITH [
{id: '3fa85f64-5717-4562-b3fc-2c963f66afa6', email: 'nguyenvana@ctu.edu.vn', studentId: 'B2106001', batch: 2021, fullName: 'Nguyễn Văn A', gender: 'Nam', bio: 'Sinh viên đam mê phát triển web', major: 'Công nghệ Phần mềm'},
{id: '3fa85f64-5717-4562-b3fc-2c963f66afa7', email: 'tranthib@ctu.edu.vn', studentId: 'B2106002', batch: 2021, fullName: 'Trần Thị B', gender: 'Nữ', bio: 'Kỹ sư phần mềm tập trung vào front-end', major: 'Công nghệ Phần mềm'},
{id: '3fa85f64-5717-4562-b3fc-2c963f66afa8', email: 'leminhc@ctu.edu.vn', studentId: 'B2106003', batch: 2021, fullName: 'Lê Minh C', gender: 'Nam', bio: 'Người đam mê AI và học máy', major: 'Khoa học Máy tính'},
{id: '3fa85f64-5717-4562-b3fc-2c963f66afa9', email: 'phamthid@ctu.edu.vn', studentId: 'B2206001', batch: 2022, fullName: 'Phạm Thị D', gender: 'Nữ', bio: 'Nhà khoa học dữ liệu', major: 'Hệ thống Thông tin'},
{id: '3fa85f64-5717-4562-b3fc-2c963f66afaa', email: 'nguyenvane@ctu.edu.vn', studentId: 'B2206002', batch: 2022, fullName: 'Nguyễn Văn E', gender: 'Nam', bio: 'Sinh viên kinh doanh tập trung vào marketing', major: 'Marketing'},
{id: '3fa85f64-5717-4562-b3fc-2c963f66afab', email: 'tranthif@ctu.edu.vn', studentId: 'B2306001', batch: 2023, fullName: 'Trần Thị F', gender: 'Nữ', bio: 'Sinh viên tài chính quan tâm đến đầu tư', major: 'Tài chính - Ngân hàng'},
{id: '3fa85f64-5717-4562-b3fc-2c963f66afac', email: 'levang@ctu.edu.vn', studentId: 'B2406001', batch: 2024, fullName: 'Lê Văn G', gender: 'Nam', bio: 'Sinh viên kỹ thuật xây dựng', major: 'Kỹ thuật Xây dựng'},
{id: '3fa85f64-5717-4562-b3fc-2c963f66afad', email: 'phamthih@ctu.edu.vn', studentId: 'B2506001', batch: 2025, fullName: 'Phạm Thị H', gender: 'Nữ', bio: 'Kỹ sư cơ khí chuyên về robot', major: 'Robot và Trí tuệ nhân tạo'}
] AS usersData
UNWIND usersData AS data
MERGE (u:User {id: data.id})
ON CREATE SET
u.email = data.email,
u.studentId = data.studentId,
u.fullName = data.fullName,
u.role = 'STUDENT',
u.bio = data.bio,
u.username = split(data.email, '@')[0],
u.isActive = true,
u.createdAt = datetime(),
u.updatedAt = datetime()
ON MATCH SET
u.fullName = data.fullName,
u.bio = data.bio,
u.username = COALESCE(u.username, split(data.email, '@')[0]),
u.isActive = COALESCE(u.isActive, true),
u.updatedAt = datetime()

WITH u, data
MATCH (b:Batch {year: toInteger(data.batch)})
MATCH (m:Major {name: data.major})
MATCH (g:Gender {name: data.gender})
MERGE (u)-[:IN_BATCH]->(b)
MERGE (u)-[:ENROLLED_IN]->(m)
MERGE (u)-[:HAS_GENDER]->(g);

// =================================================================
// 6. UPDATE EXISTING USERS WITH MISSING FIELDS (MIGRATION)
// =================================================================
MATCH (u:User)
WHERE u.username IS NULL OR u.isActive IS NULL
SET u.username = COALESCE(u.username, split(u.email, '@')[0]),
    u.isActive = COALESCE(u.isActive, true),
    u.updatedAt = datetime();

// =================================================================
// 7. FRIEND RELATIONS
// =================================================================
WITH [
['3fa85f64-5717-4562-b3fc-2c963f66afa6', '3fa85f64-5717-4562-b3fc-2c963f66afa7'],
['3fa85f64-5717-4562-b3fc-2c963f66afa6', '3fa85f64-5717-4562-b3fc-2c963f66afa8'],
['3fa85f64-5717-4562-b3fc-2c963f66afa7', '3fa85f64-5717-4562-b3fc-2c963f66afa9'],
['3fa85f64-5717-4562-b3fc-2c963f66afa8', '3fa85f64-5717-4562-b3fc-2c963f66afaa']
] AS friendships
UNWIND friendships AS pair
MATCH (u1:User {id: pair[0]})
MATCH (u2:User {id: pair[1]})
MERGE (u1)-[:IS_FRIENDS_WITH {since: datetime()}]-(u2);

// =================================================================
// 8. PENDING FRIEND REQUESTS
// =================================================================
WITH [
{sender: '3fa85f64-5717-4562-b3fc-2c963f66afab', receiver: '3fa85f64-5717-4562-b3fc-2c963f66afac'},
{sender: '3fa85f64-5717-4562-b3fc-2c963f66afac', receiver: '3fa85f64-5717-4562-b3fc-2c963f66afa6'}
] AS requests
UNWIND requests AS request
MATCH (sender:User {id: request.sender})
MATCH (receiver:User {id: request.receiver})
MERGE (sender)-[:SENT_FRIEND_REQUEST_TO {requestedAt: datetime()}]->(receiver);

// 8. CREATE POSITION, ACADEMIC, DEGREE NODES (MỚI)
// =================================================================
// --- POSITION (Chức danh/Chức vụ) ---
WITH [
    { code: 'LECTURER', name: 'Giảng viên' },
    { code: 'SNR_LECTURER', name: 'Giảng viên chính' },
    { code: 'HEAD_OF_DEPT', name: 'Trưởng bộ môn' },
    { code: 'RESEARCHER', name: 'Nghiên cứu viên' }
] AS positions
UNWIND positions AS data
MERGE (p:Position {code: data.code, name: data.name});

// --- ACADEMIC (Học hàm) ---
WITH [
    { code: 'PROF', name: 'Giáo sư' },
    { code: 'ASSOC_PROF', name: 'Phó Giáo sư' },
    { code: 'DR', name: 'Tiến sĩ' },
    { code: 'MASTER', name: 'Thạc sĩ' }
] AS academics
UNWIND academics AS data
MERGE (a:Academic {code: data.code, name: data.name});

// --- DEGREE (Văn bằng/Bằng cấp) ---
WITH [
    { code: 'BACHELOR', name: 'Cử nhân/Kỹ sư' },
    { code: 'MASTER', name: 'Thạc sĩ' },
    { code: 'PHD', name: 'Tiến sĩ' }
] AS degrees
UNWIND degrees AS data
MERGE (d:Degree {code: data.code, name: data.name});

// =================================================================
// 9. DONE
// =================================================================
RETURN '✅ Neo4j initialized with fixed UUID users, structure, and relationships.' AS status;
