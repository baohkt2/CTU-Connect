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

