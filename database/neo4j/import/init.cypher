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
CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gender) REQUIRE g.code IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Position) REQUIRE p.code IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Academic) REQUIRE a.code IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Degree) REQUIRE d.code IS UNIQUE;
// =================================================================
// 2. CREATE GENDER NODES FIRST
// =================================================================
MERGE (g1:Gender {code: 'M', name: 'Nam'})
MERGE (g2:Gender {code: 'F', name: 'Nữ'});

// =================================================================
// 3. CREATE BATCH (COURSE YEAR) NODES
// =================================================================
WITH [2021, 2022, 2023, 2024, 2025] AS batches
UNWIND batches AS batchYear
MERGE (b:Batch {year: toString(batchYear)});

// =================================================================
// 4. CREATE HIERARCHICAL UNIVERSITY STRUCTURE
// =================================================================
MERGE (uni:University {name: 'Đại học Cần Thơ', established: 1966})

WITH uni, [
{ college: 'Khoa Công nghệ Thông tin và Truyền thông', code: 'CNTT', faculties: [
{ faculty: 'Bộ môn Công nghệ Phần mềm', code: 'CNPM', majors: [
  {name: 'Công nghệ Phần mềm', code: 'CNPM01'},
  {name: 'Khoa học Máy tính', code: 'KHMT01'}
]},
{ faculty: 'Bộ môn Hệ thống Thông tin', code: 'HTTT', majors: [
  {name: 'Hệ thống Thông tin', code: 'HTTT01'},
  {name: 'Kinh doanh Kỹ thuật Số', code: 'KDKTS01'}
]},
{ faculty: 'Bộ môn Truyền thông Dữ liệu và Mạng máy tính', code: 'TTDL', majors: [
  {name: 'Mạng máy tính và Truyền thông dữ liệu', code: 'MMT01'}
]}
]},
{ college: 'Khoa Kinh tế', code: 'KT', faculties: [
{ faculty: 'Bộ môn Kinh tế học', code: 'KTH', majors: [
  {name: 'Kinh tế học', code: 'KT01'}
]},
{ faculty: 'Bộ môn Quản trị Kinh doanh', code: 'QTKD', majors: [
  {name: 'Marketing', code: 'MKT01'},
  {name: 'Quản trị Kinh doanh', code: 'QTKD01'}
]},
{ faculty: 'Bộ môn Tài chính Ngân hàng', code: 'TCNH', majors: [
  {name: 'Tài chính - Ngân hàng', code: 'TC01'}
]}
]},
{ college: 'Khoa Kỹ thuật', code: 'KYTH', faculties: [
{ faculty: 'Bộ môn Kỹ thuật Xây dựng', code: 'KTXD', majors: [
  {name: 'Kỹ thuật Xây dựng', code: 'KTXD01'}
]},
{ faculty: 'Bộ môn Kỹ thuật Cơ khí', code: 'KTCK', majors: [
  {name: 'Kỹ thuật Cơ khí', code: 'KTCK01'},
  {name: 'Robot và Trí tuệ nhân tạo', code: 'ROBOT01'}
]}
]},
{ college: 'Khoa Nông nghiệp', code: 'NN', faculties: [] },
{ college: 'Khoa Y Dược', code: 'YD', faculties: [] },
{ college: 'Khoa Sư phạm', code: 'SP', faculties: [] },
{ college: 'Khoa Khoa học Tự nhiên', code: 'KHTN', faculties: [] },
{ college: 'Khoa Khoa học Xã hội và Nhân văn', code: 'KHXHNV', faculties: [] },
{ college: 'Khoa Luật', code: 'LUAT', faculties: [] }
] AS universityStructure

UNWIND universityStructure AS collegeData
MERGE (c:College {name: collegeData.college, code: collegeData.code})
MERGE (uni)-[:HAS_COLLEGE]->(c)

WITH collegeData, c
WHERE size(collegeData.faculties) > 0
UNWIND collegeData.faculties AS facultyData
MERGE (f:Faculty {name: facultyData.faculty, code: facultyData.code, college: collegeData.college})
MERGE (c)-[:HAS_FACULTY]->(f)

WITH facultyData, f
WHERE size(facultyData.majors) > 0
UNWIND facultyData.majors AS majorData
MERGE (m:Major {name: majorData.name, code: majorData.code, faculty: facultyData.faculty})
MERGE (f)-[:HAS_MAJOR]->(m);

// =================================================================
// 5. POSITION AND DEGREE NODES
// =================================================================
UNWIND [
{code: 'GIANG_VIEN', name: 'Giảng viên'},
{code: 'GIANG_VIEN_CHINH', name: 'Giảng viên chính'},
{code: 'CAN_BO', name: 'Cán bộ'},
{code: 'TRO_LY', name: 'Trợ lý'},
{code: 'NGHIEN_CUU_VIEN', name: 'Nghiên cứu viên'}
] AS position
MERGE (p:Position {code: position.code})
SET p.name = position.name;


UNWIND [
{code: 'GIAO_SU', name: 'Giáo sư'},
{code: 'PHO_GIAO_SU', name: 'Phó Giáo sư'},
{code: 'TIEN_SI', name: 'Tiến sĩ'},
{code: 'THAC_SI', name: 'Thạc sĩ'},
{code: 'CU_NHAN', name: 'Cử nhân'}
] AS title
MERGE (a:Academic {code: title.code})
SET a.name = title.name;

UNWIND [
{code: 'TIEN_SI', name: 'Tiến sĩ'},
{code: 'THAC_SI', name: 'Thạc sĩ'},
{code: 'CU_NHAN', name: 'Cử nhân'},
{code: 'KHAC', name: 'Khác'}
] AS degree
MERGE (d:Degree {code: degree.code})
SET d.name = degree.name;

CREATE CONSTRAINT IF NOT EXISTS FOR (p:Position) REQUIRE p.code IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Academic) REQUIRE a.code IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Degree) REQUIRE d.code IS UNIQUE;

// =================================================================
// 5. CREATE SAMPLE USERS FOR TESTING
// =================================================================
// Create a sample student user

// =================================================================
// 6. CREATE RELATIONSHIPS FOR SAMPLE USERS
// =================================================================
// Student relationships
