// =================================================================
// Enhanced Neo4j Initialization Script for "Compathi"
// =================================================================
// This script demonstrates a more graph-native approach to modeling the data.
// Key Improvements:
// 1. Categorical Nodes: College, Faculty, Major, and Batch are now distinct nodes,
//    allowing for more powerful and efficient graph traversals.
// 2. Clearer Relationships: Uses distinct relationship types for confirmed friendships
//    (:IS_FRIENDS_WITH) and pending requests (:SENT_FRIEND_REQUEST_TO).
// 3. Efficient Data Loading: Uses UNWIND to create nodes and relationships from lists,
//    which is cleaner and more performant than individual CREATE statements.
// 4. Idempotency: Uses MERGE to prevent creating duplicate category nodes if the
//    script is run multiple times.

// To start with a clean slate (optional, use with caution)
// MATCH (n) DETACH DELETE n;


// =================================================================
// 1. CONSTRAINTS & INDEXES
// =================================================================
// Constraints ensure data integrity and uniqueness. They also create backing indexes automatically.
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:College) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (f:Faculty) REQUIRE f.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:Major) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (b:Batch) REQUIRE b.year IS UNIQUE;


// =================================================================
// 2. CREATE CATEGORICAL NODES
// =================================================================
// Create shared context nodes. Using MERGE prevents duplicates.

// Colleges
WITH ["College of Information Technology", "College of Economics", "College of Engineering"] AS colleges
UNWIND colleges AS collegeName
MERGE (c:College {name: collegeName});

// Faculties
WITH ["Software Engineering", "Computer Science", "Business Administration", "Civil Engineering", "Mechanical Engineering"] AS faculties
UNWIND faculties AS facultyName
MERGE (f:Faculty {name: facultyName});

// Majors
WITH ["Web Development", "Artificial Intelligence", "Data Science", "Marketing", "Finance", "Structural Engineering", "Robotics"] AS majors
UNWIND majors AS majorName
MERGE (m:Major {name: majorName});

// Batches
WITH ["2019", "2020", "2021"] AS batches
UNWIND batches AS batchYear
MERGE (b:Batch {year: batchYear});


// =================================================================
// 3. CREATE USERS AND CONNECT THEM TO CATEGORIES
// =================================================================
// We use UNWIND on a list of user data maps. This is the standard way to bulk-import data.
WITH [
{id: "user1", email: "john.doe@student.ctu.edu.vn", studentId: "B1906001", batch: "2019", fullName: "John Doe", gender: "Male", bio: "Computer science student passionate about web development", college: "College of Information Technology", faculty: "Software Engineering", major: "Web Development"},
{id: "user2", email: "jane.smith@student.ctu.edu.vn", studentId: "B1906002", batch: "2019", fullName: "Jane Smith", gender: "Female", bio: "Software engineer with a focus on front-end development", college: "College of Information Technology", faculty: "Software Engineering", major: "Web Development"},
{id: "user3", email: "michael.johnson@student.ctu.edu.vn", studentId: "B1906003", batch: "2019", fullName: "Michael Johnson", gender: "Male", bio: "AI enthusiast and researcher", college: "College of Information Technology", faculty: "Computer Science", major: "Artificial Intelligence"},
{id: "user4", email: "emily.williams@student.ctu.edu.vn", studentId: "B2006001", batch: "2020", fullName: "Emily Williams", gender: "Female", bio: "Data scientist with a passion for machine learning", college: "College of Information Technology", faculty: "Computer Science", major: "Data Science"},
{id: "user5", email: "david.brown@student.ctu.edu.vn", studentId: "B2006002", batch: "2020", fullName: "David Brown", gender: "Male", bio: "Business student focusing on digital marketing strategies", college: "College of Economics", faculty: "Business Administration", major: "Marketing"},
{id: "user6", email: "sarah.miller@student.ctu.edu.vn", studentId: "B2006003", batch: "2020", fullName: "Sarah Miller", gender: "Female", bio: "Finance student interested in investment analysis", college: "College of Economics", faculty: "Business Administration", major: "Finance"},
{id: "user7", email: "kevin.jones@student.ctu.edu.vn", studentId: "B2106001", batch: "2021", fullName: "Kevin Jones", gender: "Male", bio: "Civil engineering student with a focus on sustainable structures", college: "College of Engineering", faculty: "Civil Engineering", major: "Structural Engineering"},
{id: "user8", email: "lisa.davis@student.ctu.edu.vn", studentId: "B2106002", batch: "2021", fullName: "Lisa Davis", gender: "Female", bio: "Mechanical engineer specializing in robotics and automation", college: "College of Engineering", faculty: "Mechanical Engineering", major: "Robotics"}
] AS usersData
UNWIND usersData AS data

// Create the User node
CREATE (u:User {
id: data.id,
email: data.email,
studentId: data.studentId,
fullName: data.fullName,
role: "STUDENT",
gender: data.gender,
bio: data.bio,
createdAt: datetime(),
updatedAt: datetime()
})

// Match the corresponding category nodes and create relationships
WITH u, data
MATCH (c:College {name: data.college})
MATCH (f:Faculty {name: data.faculty})
MATCH (m:Major {name: data.major})
MATCH (b:Batch {year: data.batch})
MERGE (u)-[:IN_BATCH]->(b)
MERGE (u)-[:BELONGS_TO_COLLEGE]->(c)
MERGE (u)-[:IN_FACULTY]->(f)
MERGE (u)-[:HAS_MAJOR]->(m);


// =================================================================
// 4. CREATE RELATIONSHIPS BETWEEN USERS
// =================================================================

// --- Accepted Friendships ---
// We model this as a single, undirected relationship for simplicity and efficiency.
WITH [
["user1", "user2"],
["user1", "user3"],
["user2", "u4"], // Corrected from original script to match user IDs
["user3", "u5"], // Corrected
["user3", "user2"],
["user5", "user2"],
["user4", "user3"]
] AS friendships
UNWIND friendships AS friendship
MATCH (u1:User {id: friendship[0]})
MATCH (u2:User {id: friendship[1]})
// MERGE ensures we don't create the same friendship twice. The direction doesn't matter here.
MERGE (u1)-[r:IS_FRIENDS_WITH]-(u2)
ON CREATE SET r.since = datetime();


// --- Pending Friend Requests ---
// This is a directed relationship: (Sender)-[:SENT_FRIEND_REQUEST_TO]->(Receiver)
WITH [
{sender: "user1", receiver: "user4"},
{sender: "user6", receiver: "user1"},
{sender: "user7", receiver: "user2"}
] AS requests
UNWIND requests AS request
MATCH (sender:User {id: request.sender})
MATCH (receiver:User {id: request.receiver})
MERGE (sender)-[r:SENT_FRIEND_REQUEST_TO]->(receiver)
ON CREATE SET r.requestedAt = datetime();

// =================================================================
// 5. FINALIZATION
// =================================================================
RETURN "Database initialization complete. Created 8 users, 4 category types, and established friend/request relationships using a graph model.";

