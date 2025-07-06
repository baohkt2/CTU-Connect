// This script initializes the Neo4j database with sample user data and relationships

// Create constraints for unique IDs and emails
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE;

// Create sample users representing students from different colleges, faculties, and batches
CREATE (u1:User {
  id: "user1",
  email: "john.doe@student.ctu.edu.vn",
  studentId: "B1906001",
  batch: "2019",
  fullName: "John Doe",
  role: "STUDENT",
  college: "College of Information Technology",
  faculty: "Software Engineering",
  major: "Web Development",
  gender: "Male",
  bio: "Computer science student passionate about web development",
  createdAt: datetime(),
  updatedAt: datetime()
});

CREATE (u2:User {
  id: "user2",
  email: "jane.smith@student.ctu.edu.vn",
  studentId: "B1906002",
  batch: "2019",
  fullName: "Jane Smith",
  role: "STUDENT",
  college: "College of Information Technology",
  faculty: "Software Engineering",
  major: "Web Development",
  gender: "Female",
  bio: "Software engineer with a focus on front-end development",
  createdAt: datetime(),
  updatedAt: datetime()
});

CREATE (u3:User {
  id: "user3",
  email: "michael.johnson@student.ctu.edu.vn",
  studentId: "B1906003",
  batch: "2019",
  fullName: "Michael Johnson",
  role: "STUDENT",
  college: "College of Information Technology",
  faculty: "Computer Science",
  major: "Artificial Intelligence",
  gender: "Male",
  bio: "AI enthusiast and researcher",
  createdAt: datetime(),
  updatedAt: datetime()
});

CREATE (u4:User {
  id: "user4",
  email: "emily.williams@student.ctu.edu.vn",
  studentId: "B2006001",
  batch: "2020",
  fullName: "Emily Williams",
  role: "STUDENT",
  college: "College of Information Technology",
  faculty: "Computer Science",
  major: "Data Science",
  gender: "Female",
  bio: "Data scientist with a passion for machine learning",
  createdAt: datetime(),
  updatedAt: datetime()
});

CREATE (u5:User {
  id: "user5",
  email: "david.brown@student.ctu.edu.vn",
  studentId: "B2006002",
  batch: "2020",
  fullName: "David Brown",
  role: "STUDENT",
  college: "College of Economics",
  faculty: "Business Administration",
  major: "Marketing",
  gender: "Male",
  bio: "Business student focusing on digital marketing strategies",
  createdAt: datetime(),
  updatedAt: datetime()
});

CREATE (u6:User {
  id: "user6",
  email: "sarah.miller@student.ctu.edu.vn",
  studentId: "B2006003",
  batch: "2020",
  fullName: "Sarah Miller",
  role: "STUDENT",
  college: "College of Economics",
  faculty: "Business Administration",
  major: "Finance",
  gender: "Female",
  bio: "Finance student interested in investment analysis",
  createdAt: datetime(),
  updatedAt: datetime()
});

CREATE (u7:User {
  id: "user7",
  email: "kevin.jones@student.ctu.edu.vn",
  studentId: "B2106001",
  batch: "2021",
  fullName: "Kevin Jones",
  role: "STUDENT",
  college: "College of Engineering",
  faculty: "Civil Engineering",
  major: "Structural Engineering",
  gender: "Male",
  bio: "Civil engineering student with a focus on sustainable structures",
  createdAt: datetime(),
  updatedAt: datetime()
});

CREATE (u8:User {
  id: "user8",
  email: "lisa.davis@student.ctu.edu.vn",
  studentId: "B2106002",
  batch: "2021",
  fullName: "Lisa Davis",
  role: "STUDENT",
  college: "College of Engineering",
  faculty: "Mechanical Engineering",
  major: "Robotics",
  gender: "Female",
  bio: "Mechanical engineer specializing in robotics and automation",
  createdAt: datetime(),
  updatedAt: datetime()
});

// Create friendship relationships (bidirectional for accepted friendships, unidirectional for pending)
// Same college, same faculty, same batch friends
CREATE (u1)-[:FRIEND]->(u2);
CREATE (u2)-[:FRIEND]->(u1);

// Same college, different faculty friends
CREATE (u1)-[:FRIEND]->(u3);
CREATE (u3)-[:FRIEND]->(u1);

// Same college, different batch friends
CREATE (u2)-[:FRIEND]->(u4);
CREATE (u4)-[:FRIEND]->(u2);

// Different college friends
CREATE (u3)-[:FRIEND]->(u5);
CREATE (u5)-[:FRIEND]->(u3);

// Pending friend requests (unidirectional)
CREATE (u1)-[:FRIEND]->(u4);
CREATE (u6)-[:FRIEND]->(u1);
CREATE (u7)-[:FRIEND]->(u2);

// Create second-degree connections for mutual friend scenarios
CREATE (u3)-[:FRIEND]->(u2);
CREATE (u5)-[:FRIEND]->(u2);
CREATE (u4)-[:FRIEND]->(u3);

// Index for better query performance
CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.college);
CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.faculty);
CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.major);
CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.batch);

// Print completion message
RETURN "Database initialization complete. Created 8 users with friendship relationships.";
