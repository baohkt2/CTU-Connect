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
WITH ['K45',' K46', 'K47', 'K48', 'K49', 'K50', 'K51'] AS batches
UNWIND batches AS batchYear
MERGE (b:Batch {year: batchYear});

// =================================================================
// 4. CREATE HIERARCHICAL UNIVERSITY STRUCTURE
// =================================================================
MERGE (uni:University {name: 'Đại học Cần Thơ', established: 1966})

WITH uni, [
  { college: 'Trường Bách khoa', code: 'BK', faculties: [
    { faculty: 'Kỹ thuật Cơ khí', code: 'KTCK', majors: [
      { name: 'Kỹ thuật Cơ khí', code: 'KTCK01' }
    ]},
    { faculty: 'Kỹ thuật Xây dựng', code: 'KTXD', majors: [
      { name: 'Kỹ thuật Xây dựng', code: 'KTXD01' }
    ]},
    { faculty: 'Kỹ thuật Thủy lợi', code: 'KTTH', majors: [
      { name: 'Kỹ thuật Thủy lợi', code: 'KTTH01' }
    ]},
    { faculty: 'Tự động hóa', code: 'TDH', majors: [
      { name: 'Tự động hóa', code: 'TDH01' }
    ]}
  ]},
  { college: 'Trường Công nghệ Thông tin và Truyền thông', code: 'CNTT', faculties: [
    { faculty: 'Khoa Công nghệ Thông tin', code: 'CNTT', majors: [
      { name: 'Công nghệ Thông tin', code: 'CNTT01' }
    ]},
    { faculty: 'Khoa Công nghệ Phần mềm', code: 'CNPM', majors: [
      { name: 'Kỹ thuật Phần mềm', code: 'CNPM01' }
    ]},
    { faculty: 'Khoa Khoa học Máy tính', code: 'KHMT', majors: [
      { name: 'Khoa học Máy tính', code: 'KHMT01' }
    ]},
    { faculty: 'Khoa Truyền thông Đa phương tiện', code: 'TTDM', majors: [
      { name: 'Truyền thông Đa phương tiện', code: 'TTDM01' }
    ]},
    { faculty: 'Khoa Mạng Máy tính và Truyền thông', code: 'MMT', majors: [
      { name: 'Mạng Máy tính và Quyền Thông', code: 'MMT01' } // Lưu ý: "Quyền Thông" có thể là lỗi typo, hãy kiểm tra lại
    ]},
    { faculty: 'Khoa Hệ thống Thông tin', code: 'HTTT', majors: [
      { name: 'Hệ thống Thông tin', code: 'HTTT01' }
    ]}
  ]},
  { college: 'Trường Kinh tế', code: 'KT', faculties: [
    { faculty: 'Khoa Kế toán - Kiểm toán', code: 'KTKT', majors: [
      { name: 'Kế toán', code: 'KT01' }
    ]},
    { faculty: 'Khoa Kinh tế học', code: 'KTH', majors: [
      { name: 'Kinh tế học', code: 'KTH01' }
    ]},
    { faculty: 'Khoa Marketing', code: 'MKT', majors: [
      { name: 'Marketing', code: 'MKT01' }
    ]},
    { faculty: 'Khoa Quản trị Kinh doanh', code: 'QTKD', majors: [
      { name: 'Quản trị Kinh doanh', code: 'QTKD01' }
    ]}
  ]},
  { college: 'Trường Nông nghiệp', code: 'NN', faculties: [
    { faculty: 'Khoa Di truyền và Chọn giống Cây trồng', code: 'DTCSCT', majors: [
      { name: 'Di truyền và Chọn giống Cây trồng', code: 'DTCSCT01' }
    ]},
    { faculty: 'Khoa Sinh lý Sinh hóa', code: 'SLSH', majors: [
      { name: 'Sinh lý Sinh hóa', code: 'SLSH01' }
    ]},
    { faculty: 'Khoa Khoa học Đất', code: 'KHD', majors: [
      { name: 'Khoa học Đất', code: 'KHD01' }
    ]},
    { faculty: 'Khoa Khoa học Cây trồng', code: 'KHCT', majors: [
      { name: 'Khoa học Cây trồng', code: 'KHCT01' }
    ]},
    { faculty: 'Khoa Bảo vệ Thực vật', code: 'BVTV', majors: [
      { name: 'Bảo vệ Thực vật', code: 'BVTV01' }
    ]},
    { faculty: 'Khoa Chăn nuôi', code: 'CN', majors: [
      { name: 'Chăn nuôi', code: 'CN01' }
    ]},
    { faculty: 'Khoa Thú y', code: 'TY', majors: [
      { name: 'Thú y', code: 'TY01' }
    ]}
  ]},
  { college: 'Trường Sư phạm', code: 'SP', faculties: [
    { faculty: 'Khoa Sư phạm Toán và Tin học', code: 'SPTT', majors: [
      { name: 'Sư phạm Toán và Tin học', code: 'SPTT01' }
    ]},
    { faculty: 'Khoa Sư phạm Vật lý', code: 'SPVL', majors: [
      { name: 'Sư phạm Vật lý', code: 'SPVL01' }
    ]},
    { faculty: 'Khoa Sư phạm Hóa học', code: 'SPHH', majors: [
      { name: 'Sư phạm Hóa học', code: 'SPHH01' }
    ]},
    { faculty: 'Khoa Sư phạm Sinh học', code: 'SPSH', majors: [
      { name: 'Sư phạm Sinh học', code: 'SPSH01' }
    ]},
    { faculty: 'Khoa Sư phạm Ngữ văn', code: 'SPNV', majors: [
      { name: 'Sư phạm Ngữ văn', code: 'SPNV01' }
    ]},
    { faculty: 'Khoa Sư phạm Lịch sử và Địa lý', code: 'SPLSDD', majors: [
      { name: 'Sư phạm Lịch sử và Địa lý', code: 'SPLSDD01' }
    ]}
  ]}
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
