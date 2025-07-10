// =================================================================
// MERGED & ENHANCED NEO4J INITIALIZATION SCRIPT
// Author: Grok 3
// Date: 2025-07-10
// Description: Initializes the hierarchical structure for Can Tho University (CTU)
// in Neo4j, including University, Colleges, Faculties, Majors, Batches, and Genders.
// Optimized for CTU Connect project, with user creation handled dynamically via Kafka.
// Uses graph-native best practices (MERGE, UNWIND, Constraints).
// =================================================================

// -----------------------------------------------------------------
// 0. CLEAN SLATE (Optional: Use with caution to reset the database)
// -----------------------------------------------------------------
// MATCH (n) DETACH DELETE n;

// =================================================================
// 1. CONSTRAINTS & INDEXES
// =================================================================
// Ensure data integrity and uniqueness for all relevant nodes
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
// Create the University node and its hierarchy (Colleges, Faculties, Majors)
MERGE (uni:University {name: 'Đại học Cần Thơ', established: 1966})

// Define the university structure as a list of maps
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

// Create Colleges and link to University
UNWIND universityStructure AS collegeData
MERGE (c:College {name: collegeData.college})
MERGE (uni)-[:HAS_COLLEGE]->(c)

// Create Faculties and link to their College
WITH collegeData, c
WHERE size(collegeData.faculties) > 0
UNWIND collegeData.faculties AS facultyData
MERGE (f:Faculty {name: facultyData.faculty, college: collegeData.college})
MERGE (c)-[:HAS_FACULTY]->(f)

// Create Majors and link to their Faculty
WITH facultyData, f
UNWIND facultyData.majors AS majorName
MERGE (m:Major {name: majorName, faculty: facultyData.faculty})
MERGE (f)-[:HAS_MAJOR]->(m);

// =================================================================
// 3. CREATE BATCH (COURSE YEAR) NODES
// =================================================================
// Create Batch nodes for years 2021–2025
WITH [2021, 2022, 2023, 2024, 2025] AS batches
MATCH (uni:University {name: 'Đại học Cần Thơ'})
UNWIND batches AS batchYear
MERGE (b:Batch {year: toInteger(batchYear)})
MERGE (uni)-[:HAS_BATCH]->(b);

// =================================================================
// 4. CREATE GENDER NODES
// =================================================================
// Create Gender nodes for Male and Female
MERGE (g1:Gender {name: 'Nam'})
MERGE (g2:Gender {name: 'Nữ'});

// =================================================================
// 5. CREATE SAMPLE USERS AND CONNECT TO STRUCTURE
// =================================================================
// Sample users to demonstrate structure (dynamic user creation handled by user-service via Kafka)
WITH [
{id: 'user1', email: 'nguyenvana@ctu.edu.vn', studentId: 'B2106001', batch: 2021, fullName: 'Nguyễn Văn A', gender: 'Nam', bio: 'Sinh viên đam mê phát triển web', major: 'Công nghệ Phần mềm'},
{id: 'user2', email: 'tranthib@ctu.edu.vn', studentId: 'B2106002', batch: 2021, fullName: 'Trần Thị B', gender: 'Nữ', bio: 'Kỹ sư phần mềm tập trung vào front-end', major: 'Công nghệ Phần mềm'},
{id: 'user3', email: 'leminhc@ctu.edu.vn', studentId: 'B2106003', batch: 2021, fullName: 'Lê Minh C', gender: 'Nam', bio: 'Người đam mê AI và học máy', major: 'Khoa học Máy tính'},
{id: 'user4', email: 'phamthid@ctu.edu.vn', studentId: 'B2206001', batch: 2022, fullName: 'Phạm Thị D', gender: 'Nữ', bio: 'Nhà khoa học dữ liệu', major: 'Hệ thống Thông tin'},
{id: 'user5', email: 'nguyenvane@ctu.edu.vn', studentId: 'B2206002', batch: 2022, fullName: 'Nguyễn Văn E', gender: 'Nam', bio: 'Sinh viên kinh doanh tập trung vào marketing', major: 'Marketing'},
{id: 'user6', email: 'tranthif@ctu.edu.vn', studentId: 'B2306001', batch: 2023, fullName: 'Trần Thị F', gender: 'Nữ', bio: 'Sinh viên tài chính quan tâm đến đầu tư', major: 'Tài chính - Ngân hàng'},
{id: 'user7', email: 'levang@ctu.edu.vn', studentId: 'B2406001', batch: 2024, fullName: 'Lê Văn G', gender: 'Nam', bio: 'Sinh viên kỹ thuật xây dựng', major: 'Kỹ thuật Xây dựng'},
{id: 'user8', email: 'phamthih@ctu.edu.vn', studentId: 'B2506001', batch: 2025, fullName: 'Phạm Thị H', gender: 'Nữ', bio: 'Kỹ sư cơ khí chuyên về robot', major: 'Robot và Trí tuệ nhân tạo'}
] AS usersData
UNWIND usersData AS data

// Create User node
MERGE (u:User {id: data.id, email: data.email})
ON CREATE SET
u.studentId = data.studentId,
u.fullName = data.fullName,
u.role = 'STUDENT',
u.bio = data.bio,
u.createdAt = datetime(),
u.updatedAt = datetime()
ON MATCH SET
u.fullName = data.fullName,
u.bio = data.bio,
u.updatedAt = datetime()

// Link User to Batch, Major, and Gender
WITH u, data
MATCH (b:Batch {year: toInteger(data.batch)})
MATCH (m:Major {name: data.major})
MATCH (g:Gender {name: data.gender})
MERGE (u)-[:IN_BATCH]->(b)
MERGE (u)-[:ENROLLED_IN]->(m)
MERGE (u)-[:HAS_GENDER]->(g);

// =================================================================
// 6. CREATE RELATIONSHIPS BETWEEN USERS
// =================================================================
// Create accepted friendships
WITH [
['user1', 'user2'],
['user1', 'user3'],
['user2', 'user4'],
['user3', 'user5'],
['user4', 'user5'],
['user6', 'user7']
] AS friendships
UNWIND friendships AS friendship
MATCH (u1:User {id: friendship[0]})
MATCH (u2:User {id: friendship[1]})
MERGE (u1)-[r:IS_FRIENDS_WITH {since: datetime()}]->(u2);

// Create pending friend requests
WITH [
{sender: 'user1', receiver: 'user4'},
{sender: 'user6', receiver: 'user1'},
{sender: 'user7', receiver: 'user2'}
] AS requests
UNWIND requests AS request
MATCH (sender:User {id: request.sender})
MATCH (receiver:User {id: request.receiver})
MERGE (sender)-[r:SENT_FRIEND_REQUEST_TO {requestedAt: datetime()}]->(receiver);

// =================================================================
// 7. FINALIZATION
// =================================================================
RETURN 'Database initialization complete. Created CTU structure with 9 colleges, faculties, majors, batches (2021-2025), gender nodes, 8 sample users, and relationships.' AS status;