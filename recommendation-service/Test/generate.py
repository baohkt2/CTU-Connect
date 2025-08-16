import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
import uuid


class CTUConnectDataGenerator:
    """Tạo dataset mẫu cho hệ thống CTU Connect dựa trên cấu trúc thực tế"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # Danh sách khoa thực tế tại CTU (dựa trên FacultyEntity)
        self.faculties = [
            {'code': 'FIT', 'name': 'Khoa Công nghệ Thông tin & Truyền thông', 'college': 'CTU'},
            {'code': 'AGR', 'name': 'Khoa Nông nghiệp', 'college': 'CTU'},
            {'code': 'ENG', 'name': 'Khoa Kỹ thuật Công nghệ', 'college': 'CTU'},
            {'code': 'ECO', 'name': 'Khoa Kinh tế', 'college': 'CTU'},
            {'code': 'EDU', 'name': 'Khoa Sư phạm', 'college': 'CTU'},
            {'code': 'MED', 'name': 'Khoa Y Dược', 'college': 'CTU'},
            {'code': 'SCI', 'name': 'Khoa Khoa học Tự nhiên', 'college': 'CTU'},
            {'code': 'ENV', 'name': 'Khoa Môi trường & Tài nguyên Thiên nhiên', 'college': 'CTU'},
            {'code': 'AQU', 'name': 'Khoa Thủy sản', 'college': 'CTU'},
            {'code': 'LAW', 'name': 'Khoa Luật', 'college': 'CTU'},
            {'code': 'FL', 'name': 'Khoa Ngoại ngữ', 'college': 'CTU'},
            {'code': 'HUM', 'name': 'Khoa Nhân văn & Xã hội', 'college': 'CTU'}
        ]

        # Danh sách ngành học theo từng khoa (dựa trên MajorEntity)
        self.majors_by_faculty = {
            'FIT': [
                {'code': 'IT', 'name': 'Công nghệ Thông tin'},
                {'code': 'SE', 'name': 'Kỹ thuật Phần mềm'},
                {'code': 'IS', 'name': 'Hệ thống Thông tin'},
                {'code': 'AI', 'name': 'Trí tuệ Nhân tạo'},
                {'code': 'CS', 'name': 'Khoa học Máy tính'}
            ],
            'AGR': [
                {'code': 'AGRO', 'name': 'Nông học'},
                {'code': 'PLANT', 'name': 'Bảo vệ Thực vật'},
                {'code': 'SOIL', 'name': 'Khoa học Đất'},
                {'code': 'CROP', 'name': 'Khoa học Cây trồng'}
            ],
            'ENG': [
                {'code': 'CE', 'name': 'Kỹ thuật Xây dựng'},
                {'code': 'ME', 'name': 'Kỹ thuật Cơ khí'},
                {'code': 'EE', 'name': 'Kỹ thuật Điện'},
                {'code': 'ChE', 'name': 'Kỹ thuật Hóa học'}
            ],
            'ECO': [
                {'code': 'ECON', 'name': 'Kinh tế'},
                {'code': 'BIZ', 'name': 'Quản trị Kinh doanh'},
                {'code': 'ACC', 'name': 'Kế toán'},
                {'code': 'FIN', 'name': 'Tài chính Ngân hàng'}
            ],
            'EDU': [
                {'code': 'MATH_EDU', 'name': 'Sư phạm Toán'},
                {'code': 'PHYS_EDU', 'name': 'Sư phạm Vật lý'},
                {'code': 'CHEM_EDU', 'name': 'Sư phạm Hóa học'},
                {'code': 'BIO_EDU', 'name': 'Sư phạm Sinh học'}
            ]
        }

        # Vai trò người dùng (dựa trên Role enum)
        self.roles = ['STUDENT', 'LECTURER', 'STAFF', 'ADMIN', 'RESEARCHER']

        # Trình độ học vấn (dựa trên DegreeEntity)
        self.degrees = [
            {'code': 'HS', 'name': 'Tốt nghiệp THPT'},
            {'code': 'BACHELOR', 'name': 'Cử nhân'},
            {'code': 'MASTER', 'name': 'Thạc sĩ'},
            {'code': 'PHD', 'name': 'Tiến sĩ'},
            {'code': 'ASSOC_PROF', 'name': 'Phó Giáo sư'},
            {'code': 'PROF', 'name': 'Giáo sư'}
        ]

        # Chức vụ (dựa trên PositionEntity)
        self.positions = [
            {'code': 'STUDENT', 'name': 'Sinh viên'},
            {'code': 'LECTURER', 'name': 'Giảng viên'},
            {'code': 'SENIOR_LECTURER', 'name': 'Giảng viên chính'},
            {'code': 'ASSOC_PROF', 'name': 'Phó Giáo sư'},
            {'code': 'PROF', 'name': 'Giáo sư'},
            {'code': 'HEAD_DEPT', 'name': 'Trưởng bộ môn'},
            {'code': 'DEAN', 'name': 'Trưởng khoa'},
            {'code': 'ADMIN_STAFF', 'name': 'Cán bộ hành chính'}
        ]

        # Khóa học (dựa trên BatchEntity)
        self.batches = [
            {'id': f'K{year}', 'year': year}
            for year in range(2019, 2025)
        ]

        # Giới tính (dựa trên GenderEntity)
        self.genders = [
            {'id': 'MALE', 'name': 'Nam'},
            {'id': 'FEMALE', 'name': 'Nữ'},
            {'id': 'OTHER', 'name': 'Khác'}
        ]

        # Danh mục bài đăng (categories cho PostEntity)
        self.post_categories = [
            'ACADEMIC', 'RESEARCH', 'ANNOUNCEMENT', 'EVENT', 'SOCIAL',
            'CAREER', 'SCHOLARSHIP', 'COLLABORATION', 'DISCUSSION', 'NEWS'
        ]

        # Chế độ riêng tư (privacy cho PostEntity)
        self.privacy_settings = ['PUBLIC', 'FRIENDS', 'FACULTY', 'PRIVATE']

        # Content templates cho từng category
        self.content_templates = {
            'ACADEMIC': [
                "Nghiên cứu mới về ứng dụng AI trong nông nghiệp thông minh tại vùng ĐBSCL đang cho thấy kết quả khả quan.",
                "Báo cáo kết quả thí nghiệm về giống lúa chống mặn phù hợp với điều kiện biến đổi khí hậu.",
                "Phương pháp mới trong xử lý dữ liệu lớn áp dụng cho nghiên cứu nông nghiệp.",
                "Ứng dụng IoT trong quản lý tưới tiêu thông minh cho cây lúa."
            ],
            'RESEARCH': [
                "Dự án nghiên cứu về blockchain trong quản lý chuỗi cung ứng nông sản đang được triển khai.",
                "Kết quả nghiên cứu về tác động của biến đổi khí hậu đến năng suất lúa ĐBSCL.",
                "Nghiên cứu ứng dụng machine learning trong dự báo thời tiết nông nghiệp.",
                "Phát triển hệ thống giám sát chất lượng nước nuôi trồng thủy sản bằng cảm biến IoT."
            ],
            'ANNOUNCEMENT': [
                "Thông báo tổ chức hội thảo khoa học quốc tế về phát triển bền vững ĐBSCL.",
                "Thông báo tuyển sinh viên tham gia dự án nghiên cứu khoa học cấp trường.",
                "Thông báo lịch bảo vệ luận văn thạc sĩ học kỳ 2 năm 2024.",
                "Thông báo mở đăng ký học bổng Erasmus+ cho sinh viên CTU."
            ],
            'EVENT': [
                "Sự kiện giao lưu văn hóa giữa sinh viên quốc tế và sinh viên CTU.",
                "Workshop về kỹ năng nghiên cứu khoa học cho sinh viên năm cuối.",
                "Hội thảo về cơ hội việc làm trong lĩnh vực công nghệ thông tin.",
                "Ngày hội khởi nghiệp CTU 2024 - Kết nối ý tưởng, khởi tạo tương lai."
            ]
        }

    def generate_uuid(self):
        """Tạo UUID theo format của hệ thống"""
        return str(uuid.uuid4())

    def generate_users(self, n_users=100):
        """Tạo dữ liệu users theo cấu trúc UserEntity thực tế"""
        users = []

        for i in range(n_users):
            user_id = self.generate_uuid()

            # Phân bố role
            if i < n_users * 0.7:  # 70% sinh viên
                role = 'STUDENT'
            elif i < n_users * 0.9:  # 20% giảng viên
                role = 'LECTURER'
            elif i < n_users * 0.95:  # 5% nghiên cứu viên
                role = 'RESEARCHER'
            else:  # 5% cán bộ/admin
                role = random.choice(['STAFF', 'ADMIN'])

            # Tạo thông tin cơ bản
            username = f"user{i+1:04d}"
            email = f"{username}@ctu.edu.vn"
            full_name = self.generate_vietnamese_name()

            # Chọn faculty và major
            faculty = random.choice(self.faculties)
            major_list = self.majors_by_faculty.get(faculty['code'], [])
            major = random.choice(major_list) if major_list else None

            # Thông tin học tập/công việc
            if role == 'STUDENT':
                student_id = f"B{random.randint(1900000, 2199999)}"
                staff_code = None
                batch = random.choice(self.batches)
                degree = random.choice([d for d in self.degrees if d['code'] in ['HS', 'BACHELOR']])
                position = next(p for p in self.positions if p['code'] == 'STUDENT')
            else:
                student_id = None
                staff_code = f"CB{random.randint(10000, 99999)}"
                batch = None
                degree = random.choice([d for d in self.degrees if d['code'] in ['MASTER', 'PHD', 'ASSOC_PROF', 'PROF']])
                position = random.choice([p for p in self.positions if p['code'] != 'STUDENT'])

            gender = random.choice(self.genders)

            # Profile completion và thông tin bổ sung
            is_profile_completed = random.choice([True, False])
            bio = self.generate_bio(role, faculty['name']) if is_profile_completed else None
            avatar_url = f"https://example.com/avatars/{user_id}.jpg" if random.random() > 0.3 else None

            users.append({
                'id': user_id,
                'email': email,
                'username': username,
                'full_name': full_name,
                'role': role,
                'bio': bio,
                'is_profile_completed': is_profile_completed,
                'avatar_url': avatar_url,
                'background_url': None,
                'student_id': student_id,
                'staff_code': staff_code,
                'is_active': True,
                'created_at': (datetime.now() - timedelta(days=random.randint(30, 1095))).isoformat(),
                'updated_at': datetime.now().isoformat(),
                # Relations
                'faculty_code': faculty['code'],
                'faculty_name': faculty['name'],
                'major_code': major['code'] if major else None,
                'major_name': major['name'] if major else None,
                'batch_id': batch['id'] if batch else None,
                'batch_year': batch['year'] if batch else None,
                'gender_id': gender['id'],
                'gender_name': gender['name'],
                'degree_code': degree['code'],
                'degree_name': degree['name'],
                'position_code': position['code'],
                'position_name': position['name'],
                'college_name': 'Đại học Cần Thơ'
            })

        return pd.DataFrame(users)

    def generate_vietnamese_name(self):
        """Tạo tên tiếng Việt ngẫu nhiên"""
        first_names = [
            'Nguyễn', 'Trần', 'Lê', 'Phạm', 'Huỳnh', 'Hoàng', 'Phan', 'Vũ', 'Võ', 'Đặng',
            'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Lý', 'Lương', 'Trịnh', 'Đinh', 'Tô'
        ]

        middle_names = ['Văn', 'Thị', 'Hữu', 'Minh', 'Thanh', 'Hoàng', 'Quốc', 'Đức', 'Hồng', 'Thu']

        last_names_male = [
            'Nam', 'Hùng', 'Dũng', 'Tuấn', 'Hiếu', 'Phong', 'Minh', 'Quang', 'Đức', 'Bình',
            'Long', 'Thành', 'Khang', 'Hải', 'Tân', 'Việt', 'Sơn', 'Khoa', 'Tùng', 'Kiên'
        ]

        last_names_female = [
            'Linh', 'Nga', 'Hương', 'Lan', 'Mai', 'Châu', 'Thảo', 'Hoa', 'Yến', 'Nhung',
            'Trang', 'Giang', 'Phương', 'Oanh', 'Xuân', 'Thu', 'Hà', 'Diệu', 'Khánh', 'My'
        ]

        first_name = random.choice(first_names)
        middle_name = random.choice(middle_names)

        if middle_name == 'Thị':
            last_name = random.choice(last_names_female)
        else:
            last_name = random.choice(last_names_male + last_names_female)

        return f"{first_name} {middle_name} {last_name}"

    def generate_bio(self, role, faculty_name):
        """Tạo bio phù hợp với role và faculty"""
        if role == 'STUDENT':
            bios = [
                f"Sinh viên {faculty_name}, đam mê học hỏi và nghiên cứu.",
                "Yêu thích công nghệ và muốn đóng góp cho sự phát triển của ĐBSCL.",
                "Mong muốn ứng dụng kiến thức để giải quyết các vấn đề thực tiễn.",
                "Quan tâm đến nghiên cứu khoa học và phát triển bền vững."
            ]
        else:
            bios = [
                f"Giảng viên {faculty_name}, chuyên nghiên cứu và giảng dạy.",
                "Tập trung vào nghiên cứu ứng dụng phục vụ phát triển vùng ĐBSCL.",
                "Đam mê giáo dục và truyền đạt kiến thức cho thế hệ trẻ.",
                "Nghiên cứu viên với nhiều năm kinh nghiệm trong lĩnh vực chuyên môn."
            ]

        return random.choice(bios)

    def generate_posts(self, users_df, n_posts=2000):
        """Tạo dữ liệu posts theo cấu trúc PostEntity thực tế"""
        posts = []

        for i in range(n_posts):
            post_id = self.generate_uuid()

            # Chọn author từ users_df
            author_user = users_df.sample(1).iloc[0]
            author_info = {
                'id': author_user['id'],
                'username': author_user['username'],
                'fullName': author_user['full_name'],
                'avatarUrl': author_user['avatar_url'],
                'role': author_user['role'],
                'facultyName': author_user['faculty_name']
            }

            # Tạo nội dung
            category = random.choice(self.post_categories)
            title = self.generate_post_title(category, author_user['faculty_name'])
            content = self.generate_post_content(category, author_user['role'])

            # Tags
            tags = self.generate_tags(category, author_user['faculty_name'])

            # Media
            images = []
            videos = []
            if random.random() > 0.6:  # 40% có hình ảnh
                images = [f"https://example.com/images/{self.generate_uuid()}.jpg"
                         for _ in range(random.randint(1, 3))]
            if random.random() > 0.9:  # 10% có video
                videos = [f"https://example.com/videos/{self.generate_uuid()}.mp4"]

            privacy = self.determine_privacy(category, author_user['role'])

            # Stats
            view_count = random.randint(0, 1000)
            like_count = random.randint(0, min(view_count, 200))
            comment_count = random.randint(0, min(like_count, 50))
            share_count = random.randint(0, min(like_count, 30))

            created_at = datetime.now() - timedelta(
                days=random.randint(0, 180),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            posts.append({
                'id': post_id,
                'title': title,
                'content': content,
                'author_id': author_info['id'],
                'author_username': author_info['username'],
                'author_full_name': author_info['fullName'],
                'author_avatar_url': author_info['avatarUrl'],
                'author_role': author_info['role'],
                'author_faculty_name': author_info['facultyName'],
                'images': json.dumps(images),
                'videos': json.dumps(videos),
                'tags': json.dumps(tags),
                'category': category,
                'privacy': privacy,
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'share_count': share_count,
                'created_at': created_at.isoformat(),
                'updated_at': created_at.isoformat(),
                'is_active': True
            })

        return pd.DataFrame(posts)

    def generate_post_title(self, category, faculty_name):
        """Tạo tiêu đề bài đăng"""
        titles = {
            'ACADEMIC': [
                f"Nghiên cứu mới trong lĩnh vực {faculty_name}",
                "Kết quả nghiên cứu khoa học mới nhất",
                "Phương pháp mới trong nghiên cứu",
                "Báo cáo tiến độ dự án nghiên cứu"
            ],
            'RESEARCH': [
                "Dự án nghiên cứu cần tuyển thành viên",
                "Cơ hội hợp tác nghiên cứu",
                "Kết quả thí nghiệm thú vị",
                "Tìm kiếm đối tác nghiên cứu"
            ],
            'ANNOUNCEMENT': [
                "Thông báo quan trọng từ khoa",
                "Thông báo tuyển sinh",
                "Thông báo lịch thi",
                "Thông báo học bổng"
            ],
            'EVENT': [
                "Sự kiện sắp diễn ra tại CTU",
                "Workshop chuyên môn",
                "Hội thảo khoa học",
                "Giao lưu văn hóa"
            ]
        }

        return random.choice(titles.get(category, titles['ACADEMIC']))

    def generate_post_content(self, category, author_role):
        """Tạo nội dung bài đăng"""
        base_content = random.choice(self.content_templates.get(category, self.content_templates['ACADEMIC']))

        if author_role in ['LECTURER', 'RESEARCHER']:
            intro = "Từ kinh nghiệm nghiên cứu và giảng dạy, tôi muốn chia sẻ: "
        else:
            intro = "Xin chào mọi người! "

        return f"{intro}{base_content}"

    def generate_tags(self, category, faculty_name):
        """Tạo tags cho bài đăng"""
        common_tags = ['CTU', 'ĐBSCL', 'nghiêncứu', 'giáodục']
        category_tags = {
            'ACADEMIC': ['khoa học', 'academic', 'research'],
            'RESEARCH': ['nghiên cứu', 'dự án', 'hợp tác'],
            'ANNOUNCEMENT': ['thông báo', 'quan trọng'],
            'EVENT': ['sự kiện', 'workshop', 'hội thảo']
        }

        faculty_tag = faculty_name.lower().replace(' ', '').replace('&', '')

        tags = common_tags + category_tags.get(category, []) + [faculty_tag]
        return random.sample(tags, min(len(tags), random.randint(2, 5)))

    def determine_privacy(self, category, author_role):
        """Xác định privacy setting"""
        if category in ['ANNOUNCEMENT', 'EVENT']:
            return 'PUBLIC'
        elif author_role in ['LECTURER', 'RESEARCHER']:
            return random.choice(['PUBLIC', 'FACULTY'])
        else:
            return random.choice(['PUBLIC', 'FRIENDS', 'FACULTY'])

    def generate_interactions(self, users_df, posts_df, n_interactions=10000):
        """Tạo dữ liệu tương tác dựa trên InteractionEntity"""
        interactions = []

        for _ in range(n_interactions):
            user = users_df.sample(1).iloc[0]
            post = posts_df.sample(1).iloc[0]

            # Tính xác suất tương tác dựa trên:
            # 1. Cùng faculty
            # 2. Category phù hợp với role
            # 3. Privacy setting

            base_prob = 0.1

            # Cùng faculty tăng xác suất
            if user['faculty_code'] == post['author_faculty_name']:
                base_prob += 0.3

            # Role phù hợp với category
            if user['role'] == 'STUDENT' and post['category'] in ['EVENT', 'ANNOUNCEMENT']:
                base_prob += 0.2
            elif user['role'] in ['LECTURER', 'RESEARCHER'] and post['category'] in ['ACADEMIC', 'RESEARCH']:
                base_prob += 0.2

            # Privacy check
            if post['privacy'] == 'PRIVATE':
                base_prob = 0.01
            elif post['privacy'] == 'FACULTY' and user['faculty_code'] != post['author_faculty_name']:
                base_prob *= 0.3

            clicked = np.random.binomial(1, min(0.8, base_prob))

            if clicked:
                time_spent = np.random.exponential(180) + 30  # seconds
                scroll_depth = np.random.uniform(0.6, 1.0)
                completion_rate = np.random.uniform(0.7, 1.0)
                interaction_type = random.choice(['LIKE', 'COMMENT', 'SHARE', 'VIEW'])
            else:
                time_spent = np.random.exponential(20) + 5
                scroll_depth = np.random.uniform(0.1, 0.4)
                completion_rate = np.random.uniform(0.0, 0.3)
                interaction_type = 'VIEW'

            interaction_time = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            interactions.append({
                'user_id': user['id'],
                'post_id': post['id'],
                'interaction_type': interaction_type,
                'clicked': clicked,
                'time_spent': time_spent,
                'scroll_depth': scroll_depth,
                'completion_rate': completion_rate,
                'interaction_time': interaction_time.isoformat(),
                'device': random.choice(['mobile', 'desktop', 'tablet']),
                'session_duration': np.random.exponential(600) + 120
            })

        return pd.DataFrame(interactions)

    def create_training_dataset(self, users_df, posts_df, interactions_df):
        """Tạo dataset cuối cùng cho training model"""

        # Merge all data
        training_data = interactions_df.merge(
            users_df, left_on='user_id', right_on='id', how='left', suffixes=('', '_user')
        ).merge(
            posts_df, left_on='post_id', right_on='id', how='left', suffixes=('', '_post')
        )

        final_dataset = []

        for _, row in training_data.iterrows():
            # User features (chuẩn hóa)
            user_features = [
                1 if row['role'] == 'STUDENT' else 0,
                1 if row['role'] == 'LECTURER' else 0,
                1 if row['role'] == 'RESEARCHER' else 0,
                1 if row['role'] == 'STAFF' else 0,
                1 if row['role'] == 'ADMIN' else 0,
                hash(row['faculty_code']) % 20 / 20.0,  # faculty encoding
                hash(row['major_code']) % 30 / 30.0 if row['major_code'] else 0,  # major encoding
                1 if row['gender_id'] == 'MALE' else 0,
                1 if row['gender_id'] == 'FEMALE' else 0,
                1 if row['is_profile_completed'] else 0,
                hash(row['degree_code']) % 10 / 10.0,  # degree encoding
                hash(row['position_code']) % 15 / 15.0,  # position encoding
                # Time features
                pd.to_datetime(row['interaction_time']).hour / 24.0,
                pd.to_datetime(row['interaction_time']).weekday() / 7.0,
                # Device features
                1 if row['device'] == 'mobile' else 0,
                1 if row['device'] == 'desktop' else 0,
                1 if row['device'] == 'tablet' else 0
            ]

            # Post features
            post_features = [
                hash(row['category']) % 10 / 10.0,  # category encoding
                1 if row['privacy'] == 'PUBLIC' else 0,
                1 if row['privacy'] == 'FRIENDS' else 0,
                1 if row['privacy'] == 'FACULTY' else 0,
                1 if row['privacy'] == 'PRIVATE' else 0,
                row['view_count'] / 1000.0,  # normalized
                row['like_count'] / 200.0,   # normalized
                row['comment_count'] / 50.0,  # normalized
                row['share_count'] / 30.0,   # normalized
                len(row['content']) / 1000.0,  # content length normalized
                len(json.loads(row['tags'])) / 10.0,  # tags count normalized
                1 if json.loads(row['images']) else 0,  # has images
                1 if json.loads(row['videos']) else 0,  # has videos
                (datetime.now() - pd.to_datetime(row['created_at'])).days / 180.0  # days since created
            ]

            # Interaction features
            interaction_features = [
                row['time_spent'] / 600.0,  # normalized
                row['scroll_depth'],
                row['completion_rate'],
                row['session_duration'] / 1800.0,  # normalized
                1 if row['faculty_code'] == row['author_faculty_name'] else 0,  # same faculty
                1 if row['interaction_type'] == 'LIKE' else 0,
                1 if row['interaction_type'] == 'COMMENT' else 0,
                1 if row['interaction_type'] == 'SHARE' else 0
            ]

            final_dataset.append({
                'user_id': row['user_id'],
                'post_id': row['post_id'],
                'user_features': user_features,
                'post_features': post_features,
                'interaction_features': interaction_features,
                'label': row['clicked'],
                'timestamp': row['interaction_time'],
                # Metadata
                'user_role': row['role'],
                'post_category': row['category'],
                'user_faculty': row['faculty_name'],
                'post_author_faculty': row['author_faculty_name']
            })

        return pd.DataFrame(final_dataset)

    def generate_full_dataset(self, n_users=10, n_posts=20, n_interactions=100, save_path='data/'):
        """Tạo dataset hoàn chỉnh cho CTU Connect"""

        print("🚀 Bắt đầu tạo dataset cho CTU Connect...")

        # Create directory
        os.makedirs(save_path, exist_ok=True)

        # Generate data
        print("👥 Tạo dữ liệu users...")
        users_df = self.generate_users(n_users)

        print("📄 Tạo dữ liệu posts...")
        posts_df = self.generate_posts(users_df, n_posts)

        print("🤝 Tạo dữ liệu interactions...")
        interactions_df = self.generate_interactions(users_df, posts_df, n_interactions)

        print("📊 Tạo training dataset...")
        training_df = self.create_training_dataset(users_df, posts_df, interactions_df)

        # Save datasets
        users_df.to_csv(f'{save_path}/ctu_connect_users.csv', index=False)
        posts_df.to_csv(f'{save_path}/ctu_connect_posts.csv', index=False)
        interactions_df.to_csv(f'{save_path}/ctu_connect_interactions.csv', index=False)
        training_df.to_csv(f'{save_path}/ctu_connect_training.csv', index=False)

        # Save metadata
        metadata = {
            'dataset_info': {
                'name': 'CTU Connect Social Network Dataset',
                'description': 'Dataset for CTU Connect recommendation system',
                'faculties': self.faculties,
                'roles': self.roles,
                'post_categories': self.post_categories,
                'n_users': len(users_df),
                'n_posts': len(posts_df),
                'n_interactions': len(interactions_df),
                'created_date': datetime.now().isoformat()
            }
        }

        with open(f'{save_path}/ctu_connect_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Dataset CTU Connect đã được tạo thành công!")
        print(f"📁 Vị trí: {save_path}")
        print(f"👥 Số lượng users: {len(users_df)}")
        print(f"📄 Số lượng posts: {len(posts_df)}")
        print(f"🤝 Số lượng interactions: {len(interactions_df)}")
        print(f"📊 Training dataset: {len(training_df)} samples")

        return {
            'users': users_df,
            'posts': posts_df,
            'interactions': interactions_df,
            'training': training_df,
            'metadata': metadata
        }


# Example usage
if __name__ == "__main__":
    generator = CTUConnectDataGenerator(seed=42)

    dataset = generator.generate_full_dataset(
        n_users=10,
        n_posts=150,
        n_interactions=80,
        save_path='data/ctu_connect/'
    )
