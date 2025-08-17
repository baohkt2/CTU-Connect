import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
import uuid
from sklearn.preprocessing import LabelEncoder

class CTUConnectDataGenerator:
    """Tạo dataset mẫu cho hệ thống CTU Connect dựa trên cấu trúc thực tế"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # Danh sách khoa thực tế tại CTU
        self.faculties = [
            {'code': 'CNTT', 'name': 'Trường Công nghệ Thông tin và Truyền thông', 'college': 'CTU'},
            {'code': 'NN', 'name': 'Khoa Nông nghiệp', 'college': 'CTU'},
            {'code': 'KYTH', 'name': 'Khoa Kỹ thuật', 'college': 'CTU'},
            {'code': 'KT', 'name': 'Khoa Kinh tế', 'college': 'CTU'},
            {'code': 'SP', 'name': 'Khoa Sư phạm', 'college': 'CTU'},
            {'code': 'YD', 'name': 'Khoa Y Dược', 'college': 'CTU'},
            {'code': 'KHTN', 'name': 'Khoa Khoa học Tự nhiên', 'college': 'CTU'},
            {'code': 'KHXHNV', 'name': 'Khoa Khoa học Xã hội và Nhân văn', 'college': 'CTU'},
            {'code': 'LUAT', 'name': 'Khoa Luật', 'college': 'CTU'},
            {'code': 'NONE', 'name': 'Không liên quan học thuật', 'college': 'CTU'}
        ]

        # Danh sách ngành học theo từng khoa
        self.majors_by_faculty = {
            'CNTT': [
                {'code': 'CNPM01', 'name': 'Công nghệ Phần mềm'},
                {'code': 'KTPM01', 'name': 'Kỹ thuật Phần mềm'},
                {'code': 'HTTT01', 'name': 'Hệ thống Thông tin'},
                {'code': 'KHMT01', 'name': 'Khoa học Máy tính'},
                {'code': 'MMT01', 'name': 'Mạng máy tính và Truyền thông dữ liệu'},
                {'code': 'ATT01', 'name': 'An toàn Thông tin'}
            ],
            'NN': [
                {'code': 'AGRO', 'name': 'Nông học'},
                {'code': 'PLANT', 'name': 'Bảo vệ Thực vật'},
                {'code': 'SOIL', 'name': 'Khoa học Đất'},
                {'code': 'CROP', 'name': 'Khoa học Cây trồng'}
            ],
            'KYTH': [
                {'code': 'KTXD01', 'name': 'Kỹ thuật Xây dựng'},
                {'code': 'KTCK01', 'name': 'Kỹ thuật Cơ khí'},
                {'code': 'ROBOT01', 'name': 'Robot và Trí tuệ nhân tạo'}
            ],
            'KT': [
                {'code': 'KT01', 'name': 'Kinh tế học'},
                {'code': 'MKT01', 'name': 'Marketing'},
                {'code': 'QTKD01', 'name': 'Quản trị Kinh doanh'},
                {'code': 'TC01', 'name': 'Tài chính - Ngân hàng'}
            ],
            'SP': [
                {'code': 'MATH_EDU', 'name': 'Sư phạm Toán'},
                {'code': 'PHYS_EDU', 'name': 'Sư phạm Vật lý'},
                {'code': 'CHEM_EDU', 'name': 'Sư phạm Hóa học'},
                {'code': 'BIO_EDU', 'name': 'Sư phạm Sinh học'}
            ],
            'NONE': []
        }

        # Vai trò người dùng (bỏ RESEARCHER)
        self.roles = ['STUDENT', 'LECTURER', 'STAFF', 'ADMIN']

        # Trình độ học vấn
        self.degrees = [
            {'code': 'CU_NHAN', 'name': 'Cử nhân'},
            {'code': 'THAC_SI', 'name': 'Thạc sĩ'},
            {'code': 'TIEN_SI', 'name': 'Tiến sĩ'},
            {'code': 'PHO_GIAO_SU', 'name': 'Phó Giáo sư'},
            {'code': 'GIAO_SU', 'name': 'Giáo sư'},
            {'code': 'KHAC', 'name': 'Khác'}
        ]

        # Chức vụ
        self.positions = [
            {'code': 'STUDENT', 'name': 'Sinh viên'},
            {'code': 'GIANG_VIEN', 'name': 'Giảng viên'},
            {'code': 'GIANG_VIEN_CHINH', 'name': 'Giảng viên chính'},
            {'code': 'PHO_GIAO_SU', 'name': 'Phó Giáo sư'},
            {'code': 'GIAO_SU', 'name': 'Giáo sư'},
            {'code': 'CAN_BO', 'name': 'Cán bộ'},
            {'code': 'TRO_LY', 'name': 'Trợ lý'}
        ]

        # Khóa học
        self.batches = [
            {'id': f'K{year}', 'year': year}
            for year in range(2021, 2026)
        ]

        # Giới tính
        self.genders = [
            {'id': 'M', 'name': 'Nam'},
            {'id': 'F', 'name': 'Nữ'}
        ]

        # Danh mục bài viết mới
        self.post_categories = [
            'research', 'teaching', 'aquaculture', 'technology', 'climate',
            'student', 'events', 'discussion', 'other'
        ]

        # Chế độ riêng tư
        self.privacy_settings = ['PUBLIC', 'FRIENDS', 'FACULTY', 'PRIVATE']

        # Content templates cho từng category
        self.content_templates = {
            'research': [
                "Nghiên cứu mới về ứng dụng AI trong nông nghiệp thông minh tại vùng ĐBSCL.",
                "Kết quả nghiên cứu về tác động của biến đổi khí hậu đến năng suất lúa.",
                "Phát triển hệ thống giám sát chất lượng nước bằng IoT.",
                "Nghiên cứu blockchain trong quản lý chuỗi cung ứng nông sản."
            ],
            'teaching': [
                "Phương pháp giảng dạy tích cực trong môn Kỹ thuật Phần mềm.",
                "Chia sẻ tài liệu học tập cho sinh viên ngành Kinh tế.",
                "Kinh nghiệm tổ chức lớp học trực tuyến hiệu quả.",
                "Ứng dụng công nghệ thực tế ảo trong giảng dạy."
            ],
            'aquaculture': [
                "Kỹ thuật nuôi tôm bền vững tại ĐBSCL.",
                "Nghiên cứu cải thiện chất lượng nước trong nuôi trồng thủy sản.",
                "Ứng dụng cảm biến IoT trong quản lý ao nuôi tôm.",
                "Phương pháp mới trong phòng chống bệnh tôm."
            ],
            'technology': [
                "Ứng dụng AI trong tự động hóa nông nghiệp.",
                "Giới thiệu công nghệ 5G trong giáo dục.",
                "Phát triển ứng dụng di động cho sinh viên CTU.",
                "Tìm hiểu về robot trong sản xuất công nghiệp."
            ],
            'climate': [
                "Tác động của biến đổi khí hậu đến nông nghiệp ĐBSCL.",
                "Giải pháp giảm phát thải khí nhà kính trong nuôi trồng.",
                "Nghiên cứu về dự báo thời tiết ứng dụng AI.",
                "Chia sẻ kinh nghiệm thích ứng với mực nước biển dâng."
            ],
            'student': [
                "Câu chuyện sinh viên CTU vượt khó đạt học bổng quốc tế.",
                "Hành trình khởi nghiệp của sinh viên ngành Công nghệ Thông tin.",
                "Tips thi cuối kỳ hiệu quả cho sinh viên.",
                "Chia sẻ kinh nghiệm tham gia câu lạc bộ CTU."
            ],
            'events': [
                "Sự kiện giao lưu văn hóa sinh viên CTU 2025.",
                "Hội thảo công nghệ thông tin tại CTU.",
                "Ngày hội việc làm cho sinh viên ngành Kỹ thuật.",
                "Workshop kỹ năng mềm cho sinh viên năm nhất."
            ],
            'discussion': [
                "Thảo luận về vai trò của AI trong giáo dục đại học.",
                "Chia sẻ ý kiến về cải cách chương trình đào tạo.",
                "Trao đổi kinh nghiệm nghiên cứu khoa học sinh viên.",
                "Cùng bàn luận về giải pháp phát triển bền vững ĐBSCL."
            ],
            'other': [
                "Top 5 quán cà phê lý tưởng cho sinh viên Cần Thơ.",
                "Chia sẻ hành trình khám phá miền Tây cuối tuần.",
                "Bí kíp chụp ảnh đẹp tại chợ nổi Cái Răng.",
                "Câu chuyện vui về đời sống sinh viên CTU."
            ]
        }

    def generate_uuid(self):
        """Tạo UUID theo format của hệ thống"""
        return str(uuid.uuid4())

    def generate_users(self, n_users=100):
        """Tạo dữ liệu users theo cấu trúc UserEntity thực tế"""
        users = []
        faculty_encoder = LabelEncoder()
        faculty_codes = [f['code'] for f in self.faculties]
        faculty_encoded = faculty_encoder.fit_transform(faculty_codes)

        for i in range(n_users):
            user_id = self.generate_uuid()

            # Phân bố role (bỏ RESEARCHER)
            if i < n_users * 0.7:  # 70% sinh viên
                role = 'STUDENT'
            elif i < n_users * 0.9:  # 20% giảng viên
                role = 'LECTURER'
            else:  # 10% cán bộ/admin
                role = random.choice(['STAFF', 'ADMIN'])

            # Tạo thông tin cơ bản
            username = f"user{i+1:04d}"
            email = f"{username}@ctu.edu.vn"
            full_name = self.generate_vietnamese_name()

            # Chọn faculty và major
            faculty = random.choice([f for f in self.faculties if f['code'] != 'NONE'])
            major_list = self.majors_by_faculty.get(faculty['code'], [])
            major = random.choice(major_list) if major_list else None

            # Thông tin học tập/công việc
            if role == 'STUDENT':
                student_id = f"B{random.randint(1900000, 2199999)}"
                staff_code = None
                batch = random.choice(self.batches)
                degree = random.choice([d for d in self.degrees if d['code'] in ['CU_NHAN', 'KHAC']])
                position = next(p for p in self.positions if p['code'] == 'STUDENT')
            else:
                student_id = None
                staff_code = f"CB{random.randint(10000, 99999)}"
                batch = None
                degree = random.choice([d for d in self.degrees if d['code'] in ['THAC_SI', 'TIEN_SI', 'PHO_GIAO_SU', 'GIAO_SU']])
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
                'college_name': 'Đại học Cần Thơ',
                'user_faculty_encoded': faculty_encoded[faculty_codes.index(faculty['code'])] / max(faculty_encoded)
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
        last_name = random.choice(last_names_female if middle_name == 'Thị' else last_names_male + last_names_female)
        return f"{first_name} {middle_name} {last_name}"

    def generate_bio(self, role, faculty_name):
        """Tạo bio phù hợp với role và faculty"""
        if role == 'STUDENT':
            bios = [
                f"Sinh viên {faculty_name}, đam mê học hỏi và nghiên cứu.",
                f"Yêu thích công nghệ và muốn đóng góp cho sự phát triển của ĐBSCL.",
                f"Mong muốn ứng dụng kiến thức để giải quyết các vấn đề thực tiễn.",
                f"Quan tâm đến các hoạt động sinh viên tại CTU."
            ]
        else:
            bios = [
                f"Giảng viên {faculty_name}, chuyên giảng dạy và nghiên cứu.",
                f"Tập trung vào ứng dụng kiến thức phục vụ vùng ĐBSCL.",
                f"Đam mê giáo dục và truyền đạt kiến thức cho thế hệ trẻ.",
                f"Cán bộ {faculty_name}, hỗ trợ phát triển cộng đồng CTU."
            ]
        return random.choice(bios)

    def generate_posts(self, users_df, n_posts=2000):
        """Tạo dữ liệu posts theo cấu trúc PostEntity thực tế"""
        posts = []
        faculty_encoder = LabelEncoder()
        faculty_codes = [f['code'] for f in self.faculties]
        faculty_encoded = faculty_encoder.fit_transform(faculty_codes)

        for i in range(n_posts):
            post_id = self.generate_uuid()
            author_user = users_df.sample(1).iloc[0]
            author_info = {
                'id': author_user['id'],
                'username': author_user['username'],
                'fullName': author_user['full_name'],
                'avatarUrl': author_user['avatar_url'],
                'role': author_user['role'],
                'facultyName': 'Không liên quan học thuật' if random.random() < 0.1 else author_user['faculty_name']
            }

            # Tạo nội dung
            category = random.choice(self.post_categories)
            if category == 'other':
                author_info['facultyName'] = 'Không liên quan học thuật'

            title = self.generate_post_title(category, author_info['facultyName'])
            content = self.generate_post_content(category, author_user['role'])

            # Tags
            tags = self.generate_tags(category, author_info['facultyName'])

            # Media
            images = []
            videos = []
            if random.random() > 0.6:
                images = [f"https://example.com/images/{self.generate_uuid()}.jpg" for _ in range(random.randint(1, 3))]
            if random.random() > 0.9:
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
                'is_active': True,
                'post_author_faculty_encoded': faculty_encoded[faculty_codes.index('NONE' if author_info['facultyName'] == 'Không liên quan học thuật' else author_user['faculty_code'])] / max(faculty_encoded)
            })

        return pd.DataFrame(posts)

    def generate_post_title(self, category, faculty_name):
        """Tạo tiêu đề bài đăng"""
        titles = {
            'research': [
                f"Nghiên cứu mới trong lĩnh vực {faculty_name}",
                "Kết quả nghiên cứu khoa học mới nhất",
                "Phương pháp mới trong nghiên cứu",
                "Báo cáo tiến độ dự án nghiên cứu"
            ],
            'teaching': [
                f"Chia sẻ phương pháp giảng dạy tại {faculty_name}",
                "Tài liệu học tập mới cho sinh viên",
                "Kinh nghiệm giảng dạy trực tuyến",
                "Ứng dụng công nghệ trong giảng dạy"
            ],
            'aquaculture': [
                "Kỹ thuật nuôi trồng thủy sản bền vững",
                "Nghiên cứu chất lượng nước trong ao nuôi",
                "Ứng dụng IoT trong nuôi tôm",
                "Phòng chống bệnh trong nuôi trồng"
            ],
            'technology': [
                "Ứng dụng công nghệ mới tại CTU",
                "Giới thiệu công nghệ 5G trong giáo dục",
                "Phát triển ứng dụng di động cho sinh viên",
                "Khám phá robot trong sản xuất"
            ],
            'climate': [
                "Tác động biến đổi khí hậu tại ĐBSCL",
                "Giải pháp giảm phát thải khí nhà kính",
                "Dự báo thời tiết ứng dụng AI",
                "Thích ứng với mực nước biển dâng"
            ],
            'student': [
                "Câu chuyện sinh viên CTU vượt khó",
                "Hành trình khởi nghiệp sinh viên",
                "Tips thi cuối kỳ hiệu quả",
                "Hoạt động câu lạc bộ sinh viên"
            ],
            'events': [
                "Sự kiện giao lưu văn hóa CTU",
                "Hội thảo công nghệ thông tin",
                "Ngày hội việc làm CTU",
                "Workshop kỹ năng mềm"
            ],
            'discussion': [
                "Thảo luận về AI trong giáo dục",
                "Cải cách chương trình đào tạo",
                "Trao đổi nghiên cứu khoa học",
                "Giải pháp phát triển bền vững"
            ],
            'other': [
                "Khám phá quán cà phê đẹp ở Cần Thơ",
                "Hành trình du lịch miền Tây",
                "Bí kíp chụp ảnh chợ nổi",
                "Chuyện vui đời sinh viên"
            ]
        }
        return random.choice(titles.get(category, titles['other']))

    def generate_post_content(self, category, author_role):
        """Tạo nội dung bài đăng"""
        base_content = random.choice(self.content_templates.get(category, self.content_templates['other']))
        intro = "Xin chào mọi người! " if author_role == 'STUDENT' else "Từ kinh nghiệm giảng dạy, tôi muốn chia sẻ: "
        return f"{intro}{base_content}"

    def generate_tags(self, category, faculty_name):
        """Tạo tags cho bài đăng"""
        common_tags = ['CTU', 'ĐBSCL', 'giáodục']
        category_tags = {
            'research': ['nghiên cứu', 'khoa học', 'dự án'],
            'teaching': ['giảng dạy', 'đào tạo', 'học tập'],
            'aquaculture': ['thủy sản', 'nuôi trồng', 'nông nghiệp'],
            'technology': ['công nghệ', 'AI', 'IoT'],
            'climate': ['khí hậu', 'môi trường', 'bền vững'],
            'student': ['sinh viên', 'học tập', 'câu lạc bộ'],
            'events': ['sự kiện', 'hội thảo', 'workshop'],
            'discussion': ['thảo luận', 'trao đổi', 'học thuật'],
            'other': ['giải trí', 'lifestyle', 'Cần Thơ']
        }
        faculty_tag = 'none' if faculty_name == 'Không liên quan học thuật' else faculty_name.lower().replace(' ', '').replace('&', '')
        tags = common_tags + category_tags.get(category, []) + [faculty_tag]
        return random.sample(tags, min(len(tags), random.randint(2, 5)))

    def determine_privacy(self, category, author_role):
        """Xác định privacy setting"""
        if category in ['events', 'student', 'other']:
            return 'PUBLIC'
        elif author_role == 'LECTURER':
            return random.choice(['PUBLIC', 'FACULTY'])
        else:
            return random.choice(['PUBLIC', 'FRIENDS', 'FACULTY'])

    def generate_interactions(self, users_df, posts_df, n_interactions=10000):
        """Tạo dữ liệu tương tác dựa trên InteractionEntity"""
        interactions = []
        interaction_types = ['LIKE', 'SHARE', 'BOOKMARK', 'VIEW', 'COMMENT']

        for _ in range(n_interactions):
            user = users_df.sample(1).iloc[0]
            post = posts_df.sample(1).iloc[0]

            # Tính xác suất tương tác
            base_prob = 0.1
            if user['faculty_code'] == post['author_faculty_name'] or post['author_faculty_name'] == 'Không liên quan học thuật':
                base_prob += 0.3
            if user['role'] == 'STUDENT' and post['category'] in ['student', 'events', 'other']:
                base_prob += 0.2
            elif user['role'] == 'LECTURER' and post['category'] in ['research', 'teaching', 'discussion']:
                base_prob += 0.2
            if post['privacy'] == 'PRIVATE':
                base_prob = 0.01
            elif post['privacy'] == 'FACULTY' and user['faculty_code'] != post['author_faculty_name']:
                base_prob *= 0.3

            clicked = np.random.binomial(1, min(0.8, base_prob))

            if clicked:
                time_spent = np.random.exponential(180) + 30
                scroll_depth = np.random.uniform(0.6, 1.0)
                completion_rate = np.random.uniform(0.7, 1.0)
                interaction_type = random.choice(interaction_types)
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
        training_data = interactions_df.merge(
            users_df, left_on='user_id', right_on='id', how='left', suffixes=('', '_user')
        ).merge(
            posts_df, left_on='post_id', right_on='id', how='left', suffixes=('', '_post')
        )

        final_dataset = []

        for _, row in training_data.iterrows():
            # User features (9 chiều)
            user_features = [
                1 if row['role'] == 'STUDENT' else 0,
                1 if row['role'] == 'LECTURER' else 0,
                1 if row['role'] == 'STAFF' else 0,
                1 if row['role'] == 'ADMIN' else 0,
                1 if row['is_profile_completed'] else 0,
                pd.to_datetime(row['interaction_time']).hour / 24.0,
                pd.to_datetime(row['interaction_time']).weekday() / 7.0,
                row['user_faculty_encoded']
            ]

            # Post features (13 chiều)
            post_features = [
                hash(row['category']) % 10 / 10.0,
                1 if row['privacy'] == 'PUBLIC' else 0,
                1 if row['privacy'] == 'FRIENDS' else 0,
                1 if row['privacy'] == 'FACULTY' else 0,
                1 if row['privacy'] == 'PRIVATE' else 0,
                row['view_count'] / 1000.0,
                row['like_count'] / 200.0,
                row['comment_count'] / 50.0,
                row['share_count'] / 30.0,
                len(row['content']) / 1000.0,
                len(json.loads(row['tags'])) / 10.0,
                1 if json.loads(row['images']) else 0,
                row['post_author_faculty_encoded']
            ]

            # Interaction features (8 chiều)
            interaction_features = [
                row['time_spent'] / 600.0,
                row['scroll_depth'],
                row['completion_rate'],
                row['session_duration'] / 1800.0,
                1 if row['faculty_code'] == row['author_faculty_name'] or row['author_faculty_name'] == 'Không liên quan học thuật' else 0,
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
                'user_role': row['role'],
                'post_category': row['category'],
                'user_faculty': row['faculty_name'],
                'post_author_faculty': row['author_faculty_name'],
                'user_faculty_encoded': row['user_faculty_encoded'],
                'post_author_faculty_encoded': row['post_author_faculty_encoded']
            })

        return pd.DataFrame(final_dataset)

    def generate_full_dataset(self, n_users=10, n_posts=20, n_interactions=100, save_path='data/'):
        """Tạo dataset hoàn chỉnh cho CTU Connect"""
        print("🚀 Bắt đầu tạo dataset cho CTU Connect...")

        os.makedirs(save_path, exist_ok=True)

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

if __name__ == "__main__":
    generator = CTUConnectDataGenerator(seed=42)
    dataset = generator.generate_full_dataset(
        n_users=10,
        n_posts=150,
        n_interactions=80,
        save_path='data/ctu_connect/'
    )