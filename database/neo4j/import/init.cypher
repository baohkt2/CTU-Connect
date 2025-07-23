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
MERGE (b:Batch {year: toInteger(batchYear)});

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
// 5. CREATE SAMPLE USERS FOR TESTING
// =================================================================
// Create a sample student user
MERGE (student:User {
  id: 'user-123-student',
  email: 'student@ctu.edu.vn',
  username: 'student123',
  fullName: 'Nguyễn Văn A',
  role: 'STUDENT',
  isActive: true,
  isProfileCompleted: true,
  studentId: 'B2021001',
  createdAt: datetime(),
  updatedAt: datetime()
})

// Create a sample faculty user
MERGE (faculty:User {
  id: 'user-456-faculty',
  email: 'faculty@ctu.edu.vn',
  username: 'faculty456',
  fullName: 'TS. Trần Thị B',
  role: 'FACULTY',
  isActive: true,
  isProfileCompleted: true,
  staffCode: 'GV001',
  position: 'GIANG_VIEN',
  academicTitle: 'TIEN_SI',
  degree: 'TIEN_SI',
  createdAt: datetime(),
  updatedAt: datetime()
});

// =================================================================
// 6. CREATE RELATIONSHIPS FOR SAMPLE USERS
// =================================================================
// Student relationships
MATCH (student:User {id: 'user-123-student'})
MATCH (major:Major {name: 'Công nghệ Phần mềm'})
MATCH (batch:Batch {year: 2023})
MATCH (gender:Gender {code: 'M'})
MERGE (student)-[:ENROLLED_IN]->(major)
MERGE (student)-[:IN_BATCH]->(batch)
MERGE (student)-[:HAS_GENDER]->(gender);

// Faculty relationships
MATCH (faculty:User {id: 'user-456-faculty'})
MATCH (workingFaculty:Faculty {name: 'Bộ môn Công nghệ Phần mềm'})
MATCH (gender:Gender {code: 'F'})
MERGE (faculty)-[:WORKS_IN]->(workingFaculty)
MERGE (faculty)-[:HAS_GENDER]->(gender);

// Create friendship relationship
MATCH (student:User {id: 'user-123-student'})
MATCH (faculty:User {id: 'user-456-faculty'})
MERGE (student)-[:FRIEND]->(faculty)
MERGE (faculty)-[:FRIEND]->(student);
