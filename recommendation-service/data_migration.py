"""
Script để đưa dữ liệu từ dataset vào cơ sở dữ liệu thật
"""
import pandas as pd
import psycopg2
from pymongo import MongoClient
from neo4j import GraphDatabase
import json
import uuid
from datetime import datetime
import bcrypt
import os
from generate import CTUConnectDataGenerator

class DataMigration:
    def __init__(self):
        # Database connections
        self.postgres_auth_config = {
            'host': 'localhost',
            'port': 5433,
            'database': 'auth_db',
            'user': 'postgres',
            'password': 'postgres'
        }

        self.mongodb_config = {
            'host': 'localhost',
            'port': 27018,
            'database': 'post_db'
        }

        self.neo4j_config = {
            'uri': 'bolt://localhost:7687',  # Changed from 7474 to 7687
            'user': 'neo4j',
            'password': 'password'
        }

        # Initialize data generator
        self.generator = CTUConnectDataGenerator()

    def hash_password(self, password):
        """Mã hóa mật khẩu bằng bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def migrate_users_to_auth_db(self, users_df):
        """Đưa dữ liệu users vào PostgreSQL auth_db"""
        try:
            conn = psycopg2.connect(**self.postgres_auth_config)
            cursor = conn.cursor()

            print(f"Đang migrate {len(users_df)} users vào auth_db...")

            # Hash password một lần cho tất cả users
            hashed_password = self.hash_password("password123")

            batch_size = 50
            for i in range(0, len(users_df), batch_size):
                batch = users_df.iloc[i:i+batch_size]

                for _, user in batch.iterrows():
                    # Insert user vào bảng users
                    insert_query = """
                    INSERT INTO users (id, email, username, password, role, created_at, updated_at, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO NOTHING
                    """

                    cursor.execute(insert_query, (
                        user['id'],
                        user['email'],
                        user['username'],
                        hashed_password,  # Dùng chung password đã hash
                        user['role'],
                        datetime.now(),
                        datetime.now(),
                        True
                    ))

                conn.commit()
                print(f"Đã migrate {min(i+batch_size, len(users_df))}/{len(users_df)} users vào auth_db")

            cursor.close()
            conn.close()
            print("✅ Đã migrate users vào auth_db thành công!")

        except Exception as e:
            print(f"❌ Lỗi khi migrate users: {e}")

    def migrate_users_to_neo4j(self, users_df):
        """Đưa dữ liệu users vào Neo4j"""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )

            print(f"Đang migrate {len(users_df)} users vào Neo4j...")

            with driver.session() as session:
                # Tạo constraints và indexes
                session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
                session.run("CREATE INDEX user_faculty IF NOT EXISTS FOR (u:User) ON (u.facultyCode)")
                session.run("CREATE CONSTRAINT faculty_code IF NOT EXISTS FOR (f:Faculty) REQUIRE f.code IS UNIQUE")

                # Tạo các faculty nodes trước
                for faculty in self.generator.faculties:
                    session.run("""
                        MERGE (f:Faculty {code: $code})
                        SET f.name = $name, f.college = $college
                    """, code=faculty['code'], name=faculty['name'], college=faculty['college'])

                # Tạo user nodes và relationships
                for _, user in users_df.iterrows():
                    # Tạo User node với đúng tên trường
                    session.run("""
                        MERGE (u:User {id: $id})
                        SET u.username = $username,
                            u.fullName = $fullName,
                            u.email = $email,
                            u.role = $role,
                            u.facultyCode = $facultyCode,
                            u.facultyName = $facultyName,
                            u.majorCode = $majorCode,
                            u.majorName = $majorName,
                            u.degreeCode = $degreeCode,
                            u.degreeName = $degreeName,
                            u.positionCode = $positionCode,
                            u.positionName = $positionName,
                            u.batchId = $batchId,
                            u.genderId = $genderId,
                            u.bio = $bio,
                            u.avatarUrl = $avatarUrl,
                            u.collegeName = $collegeName,
                            u.createdAt = $createdAt,
                            u.updatedAt = $updatedAt,
                            u.isActive = $isActive
                    """,
                    id=user['id'],
                    username=user['username'],
                    fullName=user['full_name'],
                    email=user['email'],
                    role=user['role'],
                    facultyCode=user['faculty_code'],
                    facultyName=user['faculty_name'],
                    majorCode=user.get('major_code'),
                    majorName=user.get('major_name'),
                    degreeCode=user['degree_code'],
                    degreeName=user['degree_name'],
                    positionCode=user['position_code'],
                    positionName=user['position_name'],
                    batchId=user.get('batch_id'),
                    genderId=user['gender_id'],
                    bio=user.get('bio'),
                    avatarUrl=user.get('avatar_url'),
                    collegeName=user['college_name'],
                    createdAt=user['created_at'],
                    updatedAt=user['updated_at'],
                    isActive=user['is_active']
                    )

                    # Tạo relationship với Faculty
                    session.run("""
                        MATCH (u:User {id: $userId})
                        MATCH (f:Faculty {code: $facultyCode})
                        MERGE (u)-[:BELONGS_TO]->(f)
                    """, userId=user['id'], facultyCode=user['faculty_code'])

            driver.close()
            print("✅ Đã migrate users vào Neo4j thành công!")

        except Exception as e:
            print(f"❌ Lỗi khi migrate users vào Neo4j: {e}")

    def migrate_posts_to_mongodb(self, posts_df):
        """Đưa dữ liệu posts vào MongoDB"""
        try:
            client = MongoClient(f'mongodb://localhost:{self.mongodb_config["port"]}/')
            db = client[self.mongodb_config['database']]
            posts_collection = db.posts

            print(f"Đang migrate {len(posts_df)} posts vào MongoDB...")

            posts_data = []
            for _, post in posts_df.iterrows():
                post_doc = {
                    '_id': post['id'],
                    'title': post['title'],
                    'content': post['content'],
                    'author': {
                        'id': post['author_id'],
                        'username': post['author_username'],
                        'fullName': post['author_full_name'],
                        'avatarUrl': post['author_avatar_url'],
                        'role': post['author_role'],
                        'facultyName': post['author_faculty_name']
                    },
                    'images': json.loads(post['images']) if post['images'] else [],
                    'videos': json.loads(post['videos']) if post['videos'] else [],
                    'tags': json.loads(post['tags']) if post['tags'] else [],
                    'category': post['category'],
                    'privacy': post['privacy'],
                    'viewCount': post['view_count'],
                    'likeCount': post['like_count'],
                    'commentCount': post['comment_count'],
                    'shareCount': post['share_count'],
                    'createdAt': datetime.fromisoformat(post['created_at']),
                    'updatedAt': datetime.fromisoformat(post['updated_at']),
                    'isActive': post['is_active']
                }
                posts_data.append(post_doc)

            # Insert posts in batches
            batch_size = 100
            for i in range(0, len(posts_data), batch_size):
                batch = posts_data[i:i+batch_size]
                posts_collection.insert_many(batch, ordered=False)
                print(f"Đã insert {min(i+batch_size, len(posts_data))}/{len(posts_data)} posts")

            client.close()
            print("✅ Đã migrate posts vào MongoDB thành công!")

        except Exception as e:
            print(f"❌ Lỗi khi migrate posts: {e}")

    def generate_interactions(self, users_df, posts_df):
        """Tạo dữ liệu tương tác (likes, comments, follows, views, shares) cho Neo4j"""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )

            print("Đang tạo relationships tương tác trong Neo4j...")

            with driver.session() as session:
                import random

                # 1. Tạo FOLLOWS relationships (người dùng theo dõi nhau)
                print("  - Tạo relationships FOLLOWS...")
                for _, user in users_df.iterrows():
                    # Mỗi user follow 5-15 users khác (ưu tiên cùng faculty)
                    same_faculty_users = users_df[
                        (users_df['faculty_code'] == user['faculty_code']) &
                        (users_df['id'] != user['id'])
                    ]

                    # Chọn users để follow
                    if len(same_faculty_users) >= 8:
                        # Follow 8 người cùng faculty + 2-7 người ngẫu nhiên
                        same_faculty_follows = same_faculty_users.sample(8)
                        other_users = users_df[
                            (~users_df['id'].isin(same_faculty_follows['id'])) &
                            (users_df['id'] != user['id'])
                        ]
                        random_follows = other_users.sample(min(random.randint(2, 7), len(other_users)))
                        follows = pd.concat([same_faculty_follows, random_follows])
                    else:
                        # Follow tất cả cùng faculty + thêm người khác
                        other_users = users_df[
                            (users_df['faculty_code'] != user['faculty_code']) &
                            (users_df['id'] != user['id'])
                        ]
                        additional_count = max(0, 10 - len(same_faculty_users))
                        additional_follows = other_users.sample(min(additional_count, len(other_users)))
                        follows = pd.concat([same_faculty_users, additional_follows])

                    for _, followed_user in follows.iterrows():
                        session.run("""
                            MATCH (u1:User {id: $followerId})
                            MATCH (u2:User {id: $followedId})
                            MERGE (u1)-[:FOLLOWS {
                                createdAt: $createdAt,
                                isActive: true
                            }]->(u2)
                        """,
                        followerId=user['id'],
                        followedId=followed_user['id'],
                        createdAt=datetime.now()
                        )

                # 2. Tạo LIKED relationships (user like posts)
                print("  - Tạo relationships LIKED...")
                for _, user in users_df.iterrows():
                    # Mỗi user like 20-50 posts
                    num_likes = random.randint(20, 50)
                    liked_posts = posts_df.sample(min(num_likes, len(posts_df)))

                    for _, post in liked_posts.iterrows():
                        session.run("""
                            MATCH (u:User {id: $userId})
                            MERGE (p:Post {id: $postId})
                            SET p.title = $title,
                                p.content = $content,
                                p.category = $category,
                                p.authorId = $authorId
                            MERGE (u)-[:LIKED {
                                createdAt: $createdAt,
                                isActive: true
                            }]->(p)
                        """,
                        userId=user['id'],
                        postId=post['id'],
                        title=post['title'],
                        content=post['content'][:200] + "..." if len(post['content']) > 200 else post['content'],
                        category=post['category'],
                        authorId=post['author_id'],
                        createdAt=datetime.now()
                        )

                # 3. Tạo VIEWED relationships (user view posts)
                print("  - Tạo relationships VIEWED...")
                for _, user in users_df.iterrows():
                    # Mỗi user view 50-100 posts
                    num_views = random.randint(50, 100)
                    viewed_posts = posts_df.sample(min(num_views, len(posts_df)))

                    for _, post in viewed_posts.iterrows():
                        view_duration = random.randint(5, 300)  # 5 giây - 5 phút
                        session.run("""
                            MATCH (u:User {id: $userId})
                            MATCH (p:Post {id: $postId})
                            MERGE (u)-[:VIEWED {
                                createdAt: $createdAt,
                                duration: $duration,
                                isComplete: $isComplete
                            }]->(p)
                        """,
                        userId=user['id'],
                        postId=post['id'],
                        createdAt=datetime.now(),
                        duration=view_duration,
                        isComplete=view_duration > 30
                        )

                # 4. Tạo SHARED relationships (user share posts)
                print("  - Tạo relationships SHARED...")
                for _, user in users_df.iterrows():
                    # Mỗi user share 3-10 posts
                    num_shares = random.randint(3, 10)
                    shared_posts = posts_df.sample(min(num_shares, len(posts_df)))

                    for _, post in shared_posts.iterrows():
                        session.run("""
                            MATCH (u:User {id: $userId})
                            MATCH (p:Post {id: $postId})
                            MERGE (u)-[:SHARED {
                                createdAt: $createdAt,
                                platform: $platform
                            }]->(p)
                        """,
                        userId=user['id'],
                        postId=post['id'],
                        createdAt=datetime.now(),
                        platform=random.choice(['internal', 'facebook', 'twitter', 'linkedin'])
                        )

                # 5. Tạo COMMENTED relationships với nội dung comment
                print("  - Tạo relationships COMMENTED...")
                comment_templates = [
                    "Rất hữu ích, cảm ơn bạn đã chia sẻ!",
                    "Thông tin này rất bổ ích cho nghiên cứu của mình.",
                    "Có thể chia sẻ thêm chi tiết không?",
                    "Tuyệt vời! Mình cũng đang quan tâm đến vấn đề này.",
                    "Cảm ơn bạn, đây chính là thứ mình đang tìm kiếm.",
                    "Rất hay, bạn có tài liệu tham khảo thêm không?",
                    "Ý kiến rất thú vị, mình hoàn toàn đồng ý.",
                    "Bài viết chất lượng cao, like và share!",
                    "Kinh nghiệm quý báu, cảm ơn bạn nhiều!",
                    "Đúng vậy, mình cũng đã từng trải qua điều tương tự."
                ]

                for _, user in users_df.iterrows():
                    # Mỗi user comment 5-20 posts
                    num_comments = random.randint(5, 20)
                    commented_posts = posts_df.sample(min(num_comments, len(posts_df)))

                    for _, post in commented_posts.iterrows():
                        comment_content = random.choice(comment_templates)
                        session.run("""
                            MATCH (u:User {id: $userId})
                            MATCH (p:Post {id: $postId})
                            CREATE (c:Comment {
                                id: $commentId,
                                content: $content,
                                authorId: $authorId,
                                postId: $postId,
                                createdAt: $createdAt,
                                isActive: true
                            })
                            CREATE (u)-[:AUTHORED]->(c)
                            CREATE (c)-[:BELONGS_TO]->(p)
                        """,
                        userId=user['id'],
                        postId=post['id'],
                        commentId=str(uuid.uuid4()),
                        content=comment_content,
                        authorId=user['id'],
                        createdAt=datetime.now()
                        )

                # 6. Tạo INTERESTED_IN relationships (user quan tâm đến categories)
                print("  - Tạo relationships INTERESTED_IN...")
                categories = ['research', 'teaching', 'aquaculture', 'technology', 'climate', 'student', 'events', 'discussion', 'other']
                for _, user in users_df.iterrows():
                    # Mỗi user quan tâm đến 3-6 categories
                    interested_categories = random.sample(categories, random.randint(3, 6))

                    for category in interested_categories:
                        interest_score = random.uniform(0.4, 1.0)
                        session.run("""
                            MATCH (u:User {id: $userId})
                            MERGE (c:Category {name: $category})
                            MERGE (u)-[:INTERESTED_IN {
                                score: $score,
                                createdAt: $createdAt,
                                isActive: true
                            }]->(c)
                        """,
                        userId=user['id'],
                        category=category,
                        score=interest_score,
                        createdAt=datetime.now()
                        )

                # 7. Tạo COLLABORATED relationships (users cùng làm việc/nghiên cứu)
                print("  - Tạo relationships COLLABORATED...")
                for _, user in users_df.iterrows():
                    if user['role'] in ['LECTURER', 'RESEARCHER']:
                        # Giảng viên/nghiên cứu viên có thể hợp tác với nhau
                        same_role_users = users_df[
                            (users_df['role'].isin(['LECTURER', 'RESEARCHER'])) &
                            (users_df['id'] != user['id'])
                        ]

                        if len(same_role_users) > 0:
                            # Mỗi người hợp tác với 1-3 người khác
                            num_collaborations = random.randint(1, min(3, len(same_role_users)))
                            collaborators = same_role_users.sample(num_collaborations)

                            for _, collaborator in collaborators.iterrows():
                                session.run("""
                                    MATCH (u1:User {id: $userId1})
                                    MATCH (u2:User {id: $userId2})
                                    MERGE (u1)-[:COLLABORATED {
                                        startDate: $startDate,
                                        projectType: $projectType,
                                        isActive: true
                                    }]->(u2)
                                    MERGE (u2)-[:COLLABORATED {
                                        startDate: $startDate,
                                        projectType: $projectType,
                                        isActive: true
                                    }]->(u1)
                                """,
                                userId1=user['id'],
                                userId2=collaborator['id'],
                                startDate=datetime.now(),
                                projectType=random.choice(['research', 'teaching', 'conference', 'publication'])
                                )

            driver.close()
            print("✅ Đã tạo đầy đủ relationships tương tác thành công!")
            print("   - FOLLOWS: User theo dõi user khác")
            print("   - LIKED: User like posts")
            print("   - VIEWED: User xem posts")
            print("   - SHARED: User chia sẻ posts")
            print("   - COMMENTED: User bình luận posts")
            print("   - INTERESTED_IN: User quan tâm categories")
            print("   - COLLABORATED: Users cùng hợp tác")

        except Exception as e:
            print(f"❌ Lỗi khi tạo interactions: {e}")

    def run_migration(self, n_users=100, n_posts=500):
        """Chạy toàn bộ quá trình migration"""
        print("🚀 Bắt đầu quá trình migration dữ liệu...")

        # Generate data
        print("📊 Đang tạo dữ liệu mẫu...")
        users_df = self.generator.generate_users(n_users)
        posts_df = self.generator.generate_posts(users_df, n_posts)

        print(f"✅ Đã tạo {len(users_df)} users và {len(posts_df)} posts")

        # Save to CSV for backup
        users_df.to_csv('generated_users.csv', index=False)
        posts_df.to_csv('generated_posts.csv', index=False)
        print("💾 Đã lưu backup dữ liệu vào CSV files")

        # Migrate to databases
        print("\n🔄 Bắt đầu migration vào databases...")

        # 1. Migrate users to auth_db (PostgreSQL)
        self.migrate_users_to_auth_db(users_df)

        # 2. Migrate users to Neo4j
        self.migrate_users_to_neo4j(users_df)

        # 3. Migrate posts to MongoDB
        self.migrate_posts_to_mongodb(posts_df)

        # 4. Generate interactions in Neo4j
        self.generate_interactions(users_df, posts_df)

        print("\n🎉 Hoàn thành migration dữ liệu!")
        print("📈 Dữ liệu đã được đưa vào:")
        print(f"   - PostgreSQL (auth_db): {len(users_df)} users")
        print(f"   - Neo4j: {len(users_df)} users + relationships")
        print(f"   - MongoDB (post_db): {len(posts_df)} posts")

if __name__ == "__main__":
    migration = DataMigration()
    migration.run_migration(n_users=200, n_posts=1000)
