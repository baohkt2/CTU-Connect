// Test Data for Neo4j Graph Database
// Open Neo4j Browser: http://localhost:7474
// Copy and paste this script

// Optional: Clear all existing data
// MATCH (n) DETACH DELETE n;

// Create Users
CREATE (u1:User {
  userId: '11111111-1111-1111-1111-111111111111',
  username: 'nguyen_van_a',
  displayName: 'Nguyễn Văn A',
  facultyId: 'CNTT',
  facultyName: 'Công nghệ Thông tin',
  majorId: 'KTPM',
  majorName: 'Kỹ thuật Phần mềm',
  batchId: '2021',
  joinedDate: '2021-09-01'
})

CREATE (u2:User {
  userId: '22222222-2222-2222-2222-222222222222',
  username: 'tran_thi_b',
  displayName: 'Trần Thị B',
  facultyId: 'CNTT',
  facultyName: 'Công nghệ Thông tin',
  majorId: 'KTPM',
  majorName: 'Kỹ thuật Phần mềm',
  batchId: '2021',
  joinedDate: '2021-09-01'
})

CREATE (u3:User {
  userId: '33333333-3333-3333-3333-333333333333',
  username: 'le_van_c',
  displayName: 'Lê Văn C',
  facultyId: 'CNTT',
  facultyName: 'Công nghệ Thông tin',
  majorId: 'HTTT',
  majorName: 'Hệ thống Thông tin',
  batchId: '2021',
  joinedDate: '2021-09-01'
})

CREATE (u4:User {
  userId: '44444444-4444-4444-4444-444444444444',
  username: 'pham_thi_d',
  displayName: 'Phạm Thị D',
  facultyId: 'CNTT',
  facultyName: 'Công nghệ Thông tin',
  majorId: 'KHMT',
  majorName: 'Khoa học Máy tính',
  batchId: '2022',
  joinedDate: '2022-09-01'
})

CREATE (u5:User {
  userId: '55555555-5555-5555-5555-555555555555',
  username: 'hoang_van_e',
  displayName: 'Hoàng Văn E',
  facultyId: 'CNTT',
  facultyName: 'Công nghệ Thông tin',
  majorId: 'KTPM',
  majorName: 'Kỹ thuật Phần mềm',
  batchId: '2021',
  joinedDate: '2021-09-01'
});

// Create FRIEND relationships
MATCH (u1:User {userId: '11111111-1111-1111-1111-111111111111'})
MATCH (u2:User {userId: '22222222-2222-2222-2222-222222222222'})
CREATE (u1)-[:FRIEND {since: '2021-09-01', weight: 1.0}]->(u2)
CREATE (u2)-[:FRIEND {since: '2021-09-01', weight: 1.0}]->(u1);

MATCH (u1:User {userId: '11111111-1111-1111-1111-111111111111'})
MATCH (u3:User {userId: '33333333-3333-3333-3333-333333333333'})
CREATE (u1)-[:FRIEND {since: '2021-10-15', weight: 1.0}]->(u3)
CREATE (u3)-[:FRIEND {since: '2021-10-15', weight: 1.0}]->(u1);

MATCH (u2:User {userId: '22222222-2222-2222-2222-222222222222'})
MATCH (u3:User {userId: '33333333-3333-3333-3333-333333333333'})
CREATE (u2)-[:FRIEND {since: '2021-09-20', weight: 1.0}]->(u3)
CREATE (u3)-[:FRIEND {since: '2021-09-20', weight: 1.0}]->(u2);

MATCH (u1:User {userId: '11111111-1111-1111-1111-111111111111'})
MATCH (u5:User {userId: '55555555-5555-5555-5555-555555555555'})
CREATE (u1)-[:FRIEND {since: '2021-11-01', weight: 1.0}]->(u5)
CREATE (u5)-[:FRIEND {since: '2021-11-01', weight: 1.0}]->(u1);

MATCH (u2:User {userId: '22222222-2222-2222-2222-222222222222'})
MATCH (u5:User {userId: '55555555-5555-5555-5555-555555555555'})
CREATE (u2)-[:FRIEND {since: '2021-09-15', weight: 1.0}]->(u5)
CREATE (u5)-[:FRIEND {since: '2021-09-15', weight: 1.0}]->(u2);

// Create Faculty nodes
CREATE (f1:Faculty {
  facultyId: 'CNTT',
  name: 'Công nghệ Thông tin'
});

// Link users to faculty
MATCH (u:User {facultyId: 'CNTT'})
MATCH (f:Faculty {facultyId: 'CNTT'})
CREATE (u)-[:BELONGS_TO_FACULTY]->(f);

// Create Major nodes
CREATE (m1:Major {
  majorId: 'KTPM',
  name: 'Kỹ thuật Phần mềm',
  facultyId: 'CNTT'
})
CREATE (m2:Major {
  majorId: 'HTTT',
  name: 'Hệ thống Thông tin',
  facultyId: 'CNTT'
})
CREATE (m3:Major {
  majorId: 'KHMT',
  name: 'Khoa học Máy tính',
  facultyId: 'CNTT'
});

// Link users to majors
MATCH (u1:User {userId: '11111111-1111-1111-1111-111111111111'})
MATCH (m:Major {majorId: 'KTPM'})
CREATE (u1)-[:STUDIES_MAJOR]->(m);

MATCH (u2:User {userId: '22222222-2222-2222-2222-222222222222'})
MATCH (m:Major {majorId: 'KTPM'})
CREATE (u2)-[:STUDIES_MAJOR]->(m);

MATCH (u3:User {userId: '33333333-3333-3333-3333-333333333333'})
MATCH (m:Major {majorId: 'HTTT'})
CREATE (u3)-[:STUDIES_MAJOR]->(m);

MATCH (u4:User {userId: '44444444-4444-4444-4444-444444444444'})
MATCH (m:Major {majorId: 'KHMT'})
CREATE (u4)-[:STUDIES_MAJOR]->(m);

MATCH (u5:User {userId: '55555555-5555-5555-5555-555555555555'})
MATCH (m:Major {majorId: 'KTPM'})
CREATE (u5)-[:STUDIES_MAJOR]->(m);

// Create Batch nodes
CREATE (b1:Batch {
  batchId: '2021',
  year: 2021
})
CREATE (b2:Batch {
  batchId: '2022',
  year: 2022
});

// Link users to batches
MATCH (u:User)
WHERE u.batchId = '2021'
MATCH (b:Batch {batchId: '2021'})
CREATE (u)-[:BELONGS_TO_BATCH]->(b);

MATCH (u:User {userId: '44444444-4444-4444-4444-444444444444'})
MATCH (b:Batch {batchId: '2022'})
CREATE (u)-[:BELONGS_TO_BATCH]->(b);

// Verify data
MATCH (u:User) RETURN u.username, u.displayName, u.facultyId, u.majorId, u.batchId;

// Show relationships
MATCH (u1:User)-[r:FRIEND]->(u2:User) 
RETURN u1.username as user1, u2.username as user2, r.since as friends_since;

// Show graph structure
MATCH (u:User)-[r]->(x) 
RETURN type(r) as relationship_type, count(*) as count 
ORDER BY count DESC;

// Test query: Get friends and same-major users for user A
MATCH (u:User {userId: '11111111-1111-1111-1111-111111111111'})
OPTIONAL MATCH (u)-[:FRIEND]->(friend:User)
OPTIONAL MATCH (u)-[:STUDIES_MAJOR]->(m:Major)<-[:STUDIES_MAJOR]-(sameMajor:User)
WHERE sameMajor.userId <> u.userId
RETURN 
  u.username as user,
  collect(DISTINCT friend.username) as friends,
  collect(DISTINCT sameMajor.username) as same_major_students;
