import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
import uuid
from sklearn.preprocessing import LabelEncoder
from collections import Counter

class CTUConnectDataGenerator:
    """
    Tạo dataset mẫu cho hệ thống CTU Connect.
    PHIÊN BẢN ĐẦY ĐỦ: Kết hợp Personas + Viral/Trend + University chi tiết + Nội dung động + Phản hồi tiêu cực.
    CẢI TIẾN: Mở rộng nội dung bài viết học thuật chất lượng cao/thấp, tạo nội dung bình luận logic dựa trên chất lượng bài viết.
    """

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # Personas
        self.user_personas = {
            'LURKER': {'weight': 0.40, 'post_multiplier': 0.1, 'interaction_multiplier': 0.2},
            'ENGAGER': {'weight': 0.35, 'post_multiplier': 0.8, 'interaction_multiplier': 1.5},
            'CREATOR': {'weight': 0.15, 'post_multiplier': 3.0, 'interaction_multiplier': 1.2},
            'SPECIALIST': {'weight': 0.10, 'post_multiplier': 1.2, 'interaction_multiplier': 1.0}
        }

        # Potential trends
        self.potential_trending_tags = ['AI', 'tuyendung', 'hocbong', 'doantotnghiep', 'khoinghiep']

        # University structure (đầy đủ từ bản NÂNG CẤP)
        university_structure = [
            {'college': 'Trường Bách khoa', 'code': 'BK', 'faculties': [
                {'faculty': 'Kỹ thuật Cơ khí', 'code': 'KTCK', 'majors': [{'name': 'Kỹ thuật Cơ khí', 'code': 'KTCK01'}]},
                {'faculty': 'Kỹ thuật Xây dựng', 'code': 'KTXD', 'majors': [{'name': 'Kỹ thuật Xây dựng', 'code': 'KTXD01'}]},
                {'faculty': 'Kỹ thuật Thủy lợi', 'code': 'KTTH', 'majors': [{'name': 'Kỹ thuật Thủy lợi', 'code': 'KTTH01'}]},
                {'faculty': 'Tự động hóa', 'code': 'TDH', 'majors': [{'name': 'Tự động hóa', 'code': 'TDH01'}]}
            ]},
            {'college': 'Trường Công nghệ Thông tin và Truyền thông', 'code': 'CNTT', 'faculties': [
                {'faculty': 'Khoa Công nghệ Thông tin', 'code': 'CNTT', 'majors': [{'name': 'Công nghệ Thông tin', 'code': 'CNTT01'}]},
                {'faculty': 'Khoa Công nghệ Phần mềm', 'code': 'CNPM', 'majors': [{'name': 'Kỹ thuật Phần mềm', 'code': 'CNPM01'}]},
                {'faculty': 'Khoa Khoa học Máy tính', 'code': 'KHMT', 'majors': [{'name': 'Khoa học Máy tính', 'code': 'KHMT01'}]},
                {'faculty': 'Khoa Truyền thông Đa phương tiện', 'code': 'TTDM', 'majors': [{'name': 'Truyền thông Đa phương tiện', 'code': 'TTDM01'}]},
                {'faculty': 'Khoa Mạng Máy tính và Truyền thông', 'code': 'MMT', 'majors': [{'name': 'Mạng Máy Tính và Truyền thông', 'code': 'MMT01'}]},
                {'faculty': 'Khoa Hệ thống Thông tin', 'code': 'HTTT', 'majors': [{'name': 'Hệ thống Thông tin', 'code': 'HTTT01'}]}
            ]},
            {'college': 'Trường Kinh tế', 'code': 'KT', 'faculties': [
                {'faculty': 'Khoa Kế toán - Kiểm toán', 'code': 'KTKT', 'majors': [{'name': 'Kế toán', 'code': 'KT01'}]},
                {'faculty': 'Khoa Kinh tế học', 'code': 'KTH', 'majors': [{'name': 'Kinh tế học', 'code': 'KTH01'}]},
                {'faculty': 'Khoa Marketing', 'code': 'MKT', 'majors': [{'name': 'Marketing', 'code': 'MKT01'}]},
                {'faculty': 'Khoa Quản trị Kinh doanh', 'code': 'QTKD', 'majors': [{'name': 'Quản trị Kinh doanh', 'code': 'QTKD01'}]}
            ]},
            {'college': 'Trường Nông nghiệp', 'code': 'NN', 'faculties': [
                {'faculty': 'Khoa Khoa học Cây trồng', 'code': 'KHCT', 'majors': [{'name': 'Khoa học Cây trồng', 'code': 'KHCT01'}]},
                {'faculty': 'Khoa Bảo vệ Thực vật', 'code': 'BVTV', 'majors': [{'name': 'Bảo vệ Thực vật', 'code': 'BVTV01'}]},
                {'faculty': 'Khoa Chăn nuôi', 'code': 'CN', 'majors': [{'name': 'Chăn nuôi', 'code': 'CN01'}]},
                {'faculty': 'Khoa Thú y', 'code': 'TY', 'majors': [{'name': 'Thú y', 'code': 'TY01'}]}
            ]},
            {'college': 'Trường Sư phạm', 'code': 'SP', 'faculties': [
                {'faculty': 'Khoa Sư phạm Toán và Tin học', 'code': 'SPTT', 'majors': [{'name': 'Sư phạm Toán và Tin học', 'code': 'SPTT01'}]},
                {'faculty': 'Khoa Sư phạm Vật lý', 'code': 'SPVL', 'majors': [{'name': 'Sư phạm Vật lý', 'code': 'SPVL01'}]},
                {'faculty': 'Khoa Sư phạm Hóa học', 'code': 'SPHH', 'majors': [{'name': 'Sư phạm Hóa học', 'code': 'SPHH01'}]}
            ]}
        ]

        self.faculty_to_college_map = {}
        self.faculties = []
        self.majors_by_faculty = {}
        for college in university_structure:
            for faculty in college['faculties']:
                self.faculties.append({'code': faculty['code'], 'name': faculty['faculty'], 'college': college['college']})
                self.majors_by_faculty[faculty['code']] = faculty['majors']
                self.faculty_to_college_map[faculty['code']] = college['code']
        self.faculties.append({'code': 'NONE', 'name': 'Không liên quan học thuật', 'college': 'Đại học Cần Thơ'})
        self.majors_by_faculty['NONE'] = []
        self.faculty_to_college_map['NONE'] = 'NONE'

        # Interest mapping theo college
        self.college_interest_mapping = {
            'CNTT': ['technology', 'research', 'discussion', 'career_guidance'],
            'BK': ['technology', 'research', 'career_guidance'],
            'KT': ['discussion', 'events', 'career_guidance', 'student', 'alumni_stories'],
            'NN': ['research', 'climate', 'aquaculture', 'technology'],
            'SP': ['teaching', 'student', 'discussion', 'events'],
            'NONE': ['other', 'campus_life', 'events']
        }

        # Categories
        self.post_categories = [
            'research', 'teaching', 'aquaculture', 'technology', 'climate',
            'student', 'events', 'discussion', 'other', 'career_guidance',
            'club_activities', 'campus_life', 'alumni_stories'
        ]

        # Dynamic keywords & templates
        self.dynamic_keywords = {
            'tech_topics': ['AI', 'Machine Learning', 'Blockchain', 'Cloud Computing', 'Cybersecurity', 'IoT', 'Data Science'],
            'business_fields': ['Marketing', 'Logistics', 'Nhân sự', 'Tài chính', 'Thương mại điện tử'],
            'agricultural_innovations': ['nông nghiệp thông minh', 'canh tác hữu cơ', 'giống cây trồng mới', 'thuốc trừ sâu sinh học'],
            'soft_skills': ['giao tiếp', 'làm việc nhóm', 'tư duy phản biện', 'quản lý thời gian', 'giải quyết vấn đề'],
            'event_names': ['Workshop', 'Hội thảo', 'Talkshow', 'Cuộc thi', 'Ngày hội'],
            'job_positions': ['Software Engineer', 'Data Analyst', 'Marketing Executive', 'Kế toán viên', 'Business Analyst']
        }

        self.content_templates = {
            'research': [
                "Công bố kết quả nghiên cứu về ứng dụng {tech_topics} trong {business_fields}.",
                "Dự án NCKH về {agricultural_innovations} đã đạt kết quả ban đầu khả quan.",
                "Phân tích tác động của biến đổi khí hậu đến hệ sinh thái ĐBSCL.",
                "Mô hình mới tối ưu hóa chuỗi cung ứng nông sản sạch."
            ],
            'teaching': [
                "Chia sẻ phương pháp giảng dạy mới, áp dụng 'gamification' để tăng tương tác.",
                "Tài liệu và slide bài giảng môn chuyên ngành, mời các bạn tham khảo.",
                "Tích hợp kỹ năng {soft_skills} vào chương trình đào tạo.",
                "Kinh nghiệm hướng dẫn đồ án tốt nghiệp hiệu quả."
            ],
            'technology': [
                "Cùng tìm hiểu xu hướng {tech_topics} và ứng dụng thực tế.",
                "Project nhỏ tự động hóa công việc hằng ngày, chia sẻ để mọi người tham khảo.",
                "So sánh ưu nhược điểm các nền tảng {tech_topics} phổ biến.",
                "Lộ trình trở thành {job_positions} từ con số không."
            ],
            'student': [
                "Kinh nghiệm săn học bổng thành công, mình đã chuẩn bị những gì?",
                "Review các môn học: môn nào dễ, môn nào cần đầu tư?",
                "Cân bằng học, làm thêm và hoạt động ngoại khóa như thế nào?",
                "Tips ôn thi cuối kỳ hiệu quả."
            ],
            'events': [
                "Thông báo {event_names} chủ đề '{soft_skills}' tại hội trường lớn.",
                "Đăng ký tham gia {event_names} tuyển dụng của các công ty lớn!",
                "Tổng kết {event_names} Chào tân sinh viên.",
                "Mời tham gia {event_names} giao lưu chuyên gia {business_fields}."
            ],
            'discussion': [
                "Vai trò của {tech_topics} trong giáo dục đại học tương lai?",
                "Làm gì để nâng cao ngoại ngữ cho sinh viên khối kỹ thuật?",
                "Cơ hội và thách thức sinh viên mới ra trường hiện nay.",
                "Kỹ năng {soft_skills} quan trọng nhất nhà tuyển dụng cần?"
            ],
            'career_guidance': [
                "Lộ trình sự nghiệp cho {job_positions}: cần học gì và làm gì?",
                "Phân biệt giữa các vị trí {job_positions} trong ngành.",
                "CV cần gì để gây ấn tượng với nhà tuyển dụng {business_fields}?",
                "Chia sẻ kinh nghiệm phỏng vấn thành công."
            ],
            'club_activities': [
                "CLB học thuật tuyển thành viên mới!",
                "Sinh hoạt CLB: tranh biện chủ đề {tech_topics}.",
                "Tổng kết dự án cộng đồng 'Mùa hè xanh'.",
                "Workshop về {soft_skills} hoàn toàn miễn phí."
            ],
            'campus_life': [
                "Góc check-in mới siêu đẹp ở khu 2.",
                "Review đồ ăn ở nhà ăn mới: ngon - bổ - rẻ.",
                "Tìm bạn cùng phòng trọ gần trường.",
                "Không khí sân trường mùa thi thật khác."
            ],
            'alumni_stories': [
                "Hành trình khởi nghiệp của cựu sinh viên ngành {business_fields}.",
                "Từ CTU đến vị trí quản lý tại công ty đa quốc gia.",
                "Theo đuổi đam mê {tech_topics} và câu chuyện thành công.",
                "Những kỷ niệm khó quên thời sinh viên."
            ],
            'other': [
                "Cuối tuần đi đâu ở Cần Thơ?",
                "Tổng hợp quán cà phê yên tĩnh để học bài.",
                "Chia sẻ một cuốn sách hay mình vừa đọc.",
                "Tìm đồng đội tham gia giải thể thao."
            ]
        }

        # Thêm templates cho nội dung chất lượng cao/thấp (học thuật)
        self.academic_categories = ['research', 'teaching', 'technology', 'aquaculture', 'climate']
        self.quality_levels = ['high', 'medium', 'low']
        self.quality_weights = [0.3, 0.5, 0.2]  # Xác suất cho high, medium, low

        self.high_quality_addons = [
            " Nghiên cứu dựa trên dữ liệu thực tế từ ĐBSCL, với tham chiếu từ bài báo [Reference 1]. Kết quả cho thấy cải thiện 20%.",
            " Phân tích chi tiết với mô hình toán học, công thức: y = mx + b. Thảo luận thêm về ứng dụng thực tiễn.",
            " Bao gồm biểu đồ và dữ liệu thống kê từ khảo sát 100 mẫu. Kết luận có giá trị cao cho ngành."
        ]

        self.low_quality_addons = [
            " nội dung này chỉ là ý kiến cá nhân, ko chắc chắn lắm. Có thể sai.",
            " ngắn gọn thôi, ai cần hỏi thêm thì comment.",
            " copy từ đâu đó, quên nguồn rồi. Sai chủ đề chút nhưng post tạm."
        ]

        # Templates cho bình luận
        self.comment_templates_positive = [
            "Bài viết hay quá, cảm ơn bạn đã chia sẻ!",
            "Thông tin hữu ích, mình sẽ áp dụng thử.",
            "Phân tích sâu sắc, đồng ý với quan điểm này.",
            "Hay, mong bạn viết thêm về chủ đề này."
        ]

        self.comment_templates_negative = [
            "Nội dung không chính xác, cần kiểm tra lại.",
            "Bài viết kém chất lượng, thiếu dẫn chứng.",
            "Sai chủ đề, không liên quan đến diễn đàn.",
            "Không hữu ích, lãng phí thời gian đọc."
        ]

        self.comment_templates_neutral = [
            "Cảm ơn chia sẻ.",
            "Mình có câu hỏi: ...",
            "Thú vị, nhưng cần thêm thông tin."
        ]

        # Roles etc.
        self.roles = ['STUDENT', 'LECTURER', 'STAFF', 'ADMIN']
        self.degrees = [
            {'code': 'CU_NHAN', 'name': 'Cử nhân'},
            {'code': 'THAC_SI', 'name': 'Thạc sĩ'},
            {'code': 'TIEN_SI', 'name': 'Tiến sĩ'},
            {'code': 'PHO_GIAO_SU', 'name': 'Phó Giáo sư'},
            {'code': 'GIAO_SU', 'name': 'Giáo sư'},
            {'code': 'KHAC', 'name': 'Khác'}
        ]
        self.positions = [
            {'code': 'STUDENT', 'name': 'Sinh viên'},
            {'code': 'GIANG_VIEN', 'name': 'Giảng viên'},
            {'code': 'GIANG_VIEN_CHINH', 'name': 'Giảng viên chính'},
            {'code': 'PHO_GIAO_SU', 'name': 'Phó Giáo sư'},
            {'code': 'GIAO_SU', 'name': 'Giáo sư'},
            {'code': 'CAN_BO', 'name': 'Cán bộ'},
            {'code': 'TRO_LY', 'name': 'Trợ lý'}
        ]
        self.batches = [{'id': f'K{year}', 'year': year} for year in range(2021, 2026)]
        self.genders = [{'id': 'M', 'name': 'Nam'}, {'id': 'F', 'name': 'Nữ'}]
        self.privacy_settings = ['PUBLIC', 'FRIENDS', 'FACULTY', 'PRIVATE']

    # Helpers
    def generate_uuid(self):
        return str(uuid.uuid4())

    def generate_vietnamese_name(self):
        first_names = ['Nguyễn', 'Trần', 'Lê', 'Phạm', 'Huỳnh', 'Hoàng', 'Phan', 'Vũ', 'Võ', 'Đặng', 'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương']
        middle_names = ['Văn', 'Thị', 'Hữu', 'Minh', 'Thanh', 'Hoàng', 'Quốc', 'Đức', 'Hồng', 'Thu']
        last_names_male = ['Nam', 'Hùng', 'Dũng', 'Tuấn', 'Hiếu', 'Phong', 'Minh', 'Quang', 'Đức', 'Bình', 'Long', 'Thành', 'Khang', 'Hải']
        last_names_female = ['Linh', 'Nga', 'Hương', 'Lan', 'Mai', 'Châu', 'Thảo', 'Hoa', 'Yến', 'Nhung', 'Trang', 'Giang', 'Phương', 'Oanh']
        first_name = random.choice(first_names); middle_name = random.choice(middle_names)
        last_name = random.choice(last_names_female if middle_name == 'Thị' else last_names_male + last_names_female)
        return f"{first_name} {middle_name} {last_name}"

    def generate_bio(self, role, faculty_name):
        if role == 'STUDENT':
            return random.choice([
                f"Sinh viên {faculty_name}, đam mê học hỏi và nghiên cứu.",
                "Yêu thích công nghệ và muốn đóng góp cho sự phát triển của ĐBSCL."
            ])
        else:
            return random.choice([
                f"Giảng viên {faculty_name}, chuyên giảng dạy và nghiên cứu.",
                "Tập trung vào ứng dụng kiến thức phục vụ vùng ĐBSCL."
            ])

    # Content utilities
    def generate_post_content(self, category, author_role, quality='medium'):
        template = random.choice(self.content_templates.get(category, self.content_templates['other']))
        content = template
        for key, keywords in self.dynamic_keywords.items():
            placeholder = f"{{{key}}}"
            while placeholder in content:
                content = content.replace(placeholder, random.choice(keywords), 1)
        intro = "Hello mọi người! " if author_role == 'STUDENT' else "Chào cộng đồng CTU Connect, "

        # Cải tiến cho chất lượng học thuật
        if category in self.academic_categories:
            if quality == 'high':
                addon = random.choice(self.high_quality_addons)
                content += addon
            elif quality == 'low':
                addon = random.choice(self.low_quality_addons)
                content = content[:len(content)//2] + addon  # Làm nội dung ngắn hơn
                # Thêm "lỗi" giả bằng cách thay thế ngẫu nhiên
                words = content.split()
                if len(words) > 5:
                    idx = random.randint(0, len(words)-1)
                    words[idx] = words[idx].upper()  # Giả lỗi chính tả
                content = ' '.join(words)

        return f"{intro}{content}"

    def generate_tags(self, category, faculty_name):
        common_tags = ['CTU', 'ĐBSCL', 'sinhvien']
        category_tags = {
            'research': ['nghiên cứu', 'khoa học', 'dự án'],
            'teaching': ['giảng dạy', 'đào tạo', 'học tập'],
            'technology': ['công nghệ', 'AI', 'lập trình'],
            'student': ['họctập', 'kinhnghiệm', 'họcsinh'],
            'events': ['sự kiện', 'hội thảo', 'workshop'],
            'discussion': ['thảo luận', 'trao đổi', 'ýkiến'],
            'career_guidance': ['hướng nghiệp', 'việc làm', 'CV', 'phỏng vấn'],
            'club_activities': ['câulạcbộ', 'hoạtđộng', 'độinhóm'],
            'campus_life': ['đờisống', 'campus', 'KTX'],
            'alumni_stories': ['cựusinhviên', 'thànhcông', 'khởinghiệp'],
            'other': ['giải trí', 'lifestyle', 'Cần Thơ']
        }
        faculty_tag = 'none' if faculty_name == 'Không liên quan học thuật' else faculty_name.lower().replace(' ', '').replace('&', '')
        tags = common_tags + category_tags.get(category, []) + [faculty_tag]
        # Trộn thêm khả năng tag trend
        maybe_trend = random.sample(self.potential_trending_tags, k=random.randint(0, 2))
        tags = list(set(tags + maybe_trend))
        return random.sample(tags, min(len(tags), random.randint(3, min(6, len(tags)))))

    def generate_post_title(self, category, faculty_name):
        titles = {
            'research': [f"Công bố nghiên cứu mới tại {faculty_name}", "Kết quả đột phá trong dự án NCKH"],
            'teaching': ["Phương pháp giảng dạy hiệu quả cho sinh viên", "Tài liệu học tập mới cập nhật"],
            'technology': ["Xu hướng công nghệ bạn cần biết", "Bắt đầu với AI thế nào?"],
            'student': ["Bí quyết đạt điểm A+ các môn đại cương", "Cẩm nang sống sót cho sinh viên năm nhất"],
            'events': ["Đừng bỏ lỡ sự kiện lớn tại CTU", "Hội thảo việc làm cùng doanh nghiệp hàng đầu"],
            'discussion': ["Bạn nghĩ sao về vấn đề này?", "Cùng tranh luận: Nên hay không nên?"],
            'career_guidance': ["Lộ trình sự nghiệp cho dân IT", "Cách viết CV ấn tượng trong 5 phút"],
            'club_activities': ["CLB chúng tôi có gì hot tuần này?", "Tuyển thành viên cho dự án cộng đồng"],
            'campus_life': ["Top 5 góc sống ảo tại CTU", "Review căng-tin khu 2"],
            'alumni_stories': ["Câu chuyện khởi nghiệp từ hai bàn tay trắng", "Gặp gỡ cựu sinh viên thành đạt"],
            'other': ["Cuối tuần này đi đâu chơi?", "Một ngày của sinh viên CTU"]
        }
        return random.choice(titles.get(category, titles['other']))

    def determine_privacy(self, category, author_role):
        if category in ['events', 'student', 'other', 'campus_life', 'club_activities']:
            return 'PUBLIC'
        elif author_role == 'LECTURER':
            return random.choice(['PUBLIC', 'FACULTY'])
        else:
            return random.choice(['PUBLIC', 'FRIENDS', 'FACULTY'])

    def generate_comment_content(self, quality):
        if quality == 'high':
            sentiment = random.choices(['positive', 'neutral', 'negative'], weights=[0.8, 0.15, 0.05])[0]
        elif quality == 'low':
            sentiment = random.choices(['positive', 'neutral', 'negative'], weights=[0.05, 0.15, 0.8])[0]
        else:  # medium
            sentiment = random.choices(['positive', 'neutral', 'negative'], weights=[0.4, 0.4, 0.2])[0]

        if sentiment == 'positive':
            return random.choice(self.comment_templates_positive)
        elif sentiment == 'negative':
            return random.choice(self.comment_templates_negative)
        else:
            return random.choice(self.comment_templates_neutral)

    # Generators
    def generate_users(self, n_users=100):
        users = []
        personas_list = list(self.user_personas.keys())
        weights = [p['weight'] for p in self.user_personas.values()]

        faculty_encoder = LabelEncoder()
        faculty_codes = [f['code'] for f in self.faculties]
        faculty_encoded = faculty_encoder.fit_transform(faculty_codes)

        for i in range(n_users):
            user_id = self.generate_uuid()
            persona = random.choices(personas_list, weights=weights, k=1)[0]
            role = random.choices(self.roles, weights=[0.7, 0.2, 0.05, 0.05], k=1)[0]
            username = f"user{i+1:04d}"
            email = f"{username}@ctu.edu.vn"
            full_name = self.generate_vietnamese_name()
            faculty = random.choice([f for f in self.faculties if f['code'] != 'NONE'])
            major_list = self.majors_by_faculty.get(faculty['code'], [])
            major = random.choice(major_list) if major_list else None
            if role == 'STUDENT':
                student_id = f"B{random.randint(1900000, 2199999)}"; staff_code = None; batch = random.choice(self.batches)
                degree = random.choice([d for d in self.degrees if d['code'] in ['CU_NHAN', 'KHAC']])
                position = next(p for p in self.positions if p['code'] == 'STUDENT')
            else:
                student_id = None; staff_code = f"CB{random.randint(10000, 99999)}"; batch = None
                degree = random.choice([d for d in self.degrees if d['code'] in ['THAC_SI', 'TIEN_SI', 'PHO_GIAO_SU', 'GIAO_SU']])
                position = random.choice([p for p in self.positions if p['code'] != 'STUDENT'])
            gender = random.choice(self.genders)
            is_profile_completed = random.choice([True, False])
            bio = self.generate_bio(role, faculty['name']) if is_profile_completed else None
            avatar_url = f"https://i.pravatar.cc/150?u={user_id}" if random.random() > 0.2 else None
            users.append({
                'id': user_id, 'email': email, 'username': username, 'full_name': full_name, 'role': role, 'bio': bio,
                'persona': persona,
                'is_profile_completed': is_profile_completed, 'avatar_url': avatar_url, 'background_url': None,
                'student_id': student_id, 'staff_code': staff_code, 'is_active': True,
                'created_at': (datetime.now() - timedelta(days=random.randint(30, 1095))).isoformat(),
                'updated_at': datetime.now().isoformat(), 'faculty_code': faculty['code'], 'faculty_name': faculty['name'],
                'major_code': major['code'] if major else None, 'major_name': major['name'] if major else None,
                'batch_id': batch['id'] if batch else None, 'batch_year': batch['year'] if batch else None,
                'gender_id': gender['id'], 'gender_name': gender['name'], 'degree_code': degree['code'], 'degree_name': degree['name'],
                'position_code': position['code'], 'position_name': position['name'], 'college_name': faculty['college'],
                'user_faculty_encoded': float(faculty_encoded[faculty_codes.index(faculty['code'])]) / (max(faculty_encoded) + 1e-6)
            })
        return pd.DataFrame(users)

    def generate_posts(self, users_df, n_posts=2000, trending_tags=None):
        posts = []
        creators = users_df[users_df['persona'] == 'CREATOR']
        other_users = users_df[users_df['persona'] != 'CREATOR']

        faculty_encoder = LabelEncoder(); faculty_codes = [f['code'] for f in self.faculties]
        faculty_encoded = faculty_encoder.fit_transform(faculty_codes)

        for i in range(n_posts):
            if random.random() < 0.7 and not creators.empty:
                author_user = creators.sample(1).iloc[0]
            else:
                author_user = other_users.sample(1).iloc[0]

            category = random.choice(self.post_categories)
            # 10% bài "phi học thuật"
            author_faculty_name = 'Không liên quan học thuật' if category in ['other', 'campus_life'] or random.random() < 0.1 else author_user['faculty_name']
            author_faculty_code = 'NONE' if author_faculty_name == 'Không liên quan học thuật' else author_user['faculty_code']

            # Xác định chất lượng nếu là category học thuật
            quality = random.choices(self.quality_levels, weights=self.quality_weights)[0] if category in self.academic_categories else 'medium'

            title = self.generate_post_title(category, author_faculty_name)
            content = self.generate_post_content(category, author_user['role'], quality)
            tags = self.generate_tags(category, author_faculty_name)

            # trending & viral
            is_trending_topic = any(tag in (trending_tags or []) for tag in tags)
            is_viral = random.random() < 0.02

            view_count = random.randint(50, 1000)
            if is_viral:
                view_count *= random.randint(5, 15)
            like_count = int(view_count * np.random.uniform(0.05, 0.2))
            comment_count = int(like_count * np.random.uniform(0.1, 0.4))
            share_count = int(like_count * np.random.uniform(0.05, 0.2))

            created_at = datetime.now() - timedelta(days=random.randint(0, 180), hours=random.randint(0, 23), minutes=random.randint(0, 59))

            images = [f"https://picsum.photos/seed/{self.generate_uuid()}/800/600" for _ in range(random.randint(1, 3))] if random.random() > 0.5 else []
            videos = []

            privacy = self.determine_privacy(category, author_user['role'])

            posts.append({
                'id': self.generate_uuid(),
                'title': title,
                'content': content,
                'author_id': author_user['id'],
                'author_username': author_user['username'],
                'author_full_name': author_user['full_name'],
                'author_avatar_url': author_user['avatar_url'],
                'author_role': author_user['role'],
                'author_faculty_name': author_faculty_name,
                'author_faculty_code': author_faculty_code,
                'images': json.dumps(images),
                'videos': json.dumps(videos),
                'tags': json.dumps(tags),
                'category': category,
                'quality': quality,  # Thêm chất lượng
                'privacy': privacy,
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'share_count': share_count,
                'created_at': created_at.isoformat(),
                'updated_at': created_at.isoformat(),
                'is_active': True,
                'is_trending_topic': is_trending_topic,
                'is_viral': is_viral,
                'post_author_faculty_encoded': float(faculty_encoded[faculty_codes.index(author_faculty_code)]) / (max(faculty_encoded) + 1e-6)
            })
        return pd.DataFrame(posts)

    def generate_interactions(self, users_df, posts_df, n_interactions=10000):
        interactions = []
        interaction_types_positive = ['LIKE', 'SHARE', 'COMMENT', 'BOOKMARK']

        for _ in range(n_interactions):
            user = users_df.sample(1).iloc[0]
            post = posts_df.sample(1).iloc[0]

            base_prob = 0.05
            persona_multiplier = self.user_personas[user['persona']]['interaction_multiplier']

            user_college = self.faculty_to_college_map.get(user['faculty_code'])
            interests = self.college_interest_mapping.get(user_college, [])

            if user['persona'] == 'SPECIALIST':
                if post['category'] in interests:
                    base_prob += 0.5
                else:
                    base_prob = 0.01
            else:
                if post['category'] in interests:
                    base_prob += 0.3

            if post.get('is_trending_topic', False):
                base_prob += 0.25

            # privacy effect
            if post['privacy'] == 'PRIVATE':
                base_prob = 0.01
            elif post['privacy'] == 'FACULTY' and user['faculty_code'] != post['author_faculty_code']:
                base_prob *= 0.2

            final_prob = min(0.95, base_prob * persona_multiplier)
            clicked = np.random.binomial(1, final_prob)

            if clicked:
                if user['persona'] == 'LURKER':
                    interaction_type = random.choices(['LIKE', 'BOOKMARK'], weights=[0.8, 0.2], k=1)[0]
                else:
                    interaction_type = random.choices(interaction_types_positive, weights=[0.5, 0.2, 0.2, 0.1], k=1)[0]
                label = 1
                time_spent = np.random.exponential(180) + 30
                scroll_depth = np.random.uniform(0.6, 1.0)
            else:
                # negative feedback chance
                if random.random() < 0.1:
                    interaction_type = 'HIDE_POST'
                    label = -1
                else:
                    interaction_type = 'VIEW'
                    label = 0
                time_spent = np.random.exponential(20) + 5
                scroll_depth = np.random.uniform(0.1, 0.4)

            interaction_time = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))

            interactions.append({
                'user_id': user['id'],
                'post_id': post['id'],
                'interaction_type': interaction_type,
                'label': label,
                'user_persona': user['persona'],
                'post_is_trending': bool(post['is_trending_topic']),
                'clicked': int(clicked),
                'time_spent': float(time_spent),
                'scroll_depth': float(scroll_depth),
                'completion_rate': float(scroll_depth),
                'interaction_time': interaction_time.isoformat(),
                'device': random.choice(['mobile', 'desktop', 'tablet']),
                'session_duration': float(np.random.exponential(600) + 120)
            })
        return pd.DataFrame(interactions)

    def generate_comments(self, users_df, posts_df, interactions_df):
        comments = []
        comment_interactions = interactions_df[interactions_df['interaction_type'] == 'COMMENT']

        for _, interaction in comment_interactions.iterrows():
            post = posts_df[posts_df['id'] == interaction['post_id']].iloc[0]
            user = users_df[users_df['id'] == interaction['user_id']].iloc[0]
            quality = post['quality']
            content = self.generate_comment_content(quality)
            created_at = datetime.fromisoformat(interaction['interaction_time']) + timedelta(minutes=random.randint(1, 60))

            comments.append({
                'id': self.generate_uuid(),
                'post_id': post['id'],
                'author_id': user['id'],
                'author_username': user['username'],
                'author_full_name': user['full_name'],
                'content': content,
                'like_count': random.randint(0, 10),
                'created_at': created_at.isoformat(),
                'updated_at': created_at.isoformat(),
                'is_active': True
            })

        # Cập nhật comment_count trong posts_df dựa trên số comments thực tế
        comment_counts = comment_interactions.groupby('post_id').size().reset_index(name='actual_comment_count')
        posts_df = posts_df.merge(comment_counts, left_on='id', right_on='post_id', how='left')
        posts_df['comment_count'] = posts_df['actual_comment_count'].fillna(0).astype(int)
        posts_df.drop(['post_id', 'actual_comment_count'], axis=1, inplace=True)

        return pd.DataFrame(comments), posts_df


    def create_training_dataset(self, users_df, posts_df, interactions_df):
        # Merge
        training_data = (
            interactions_df
            .merge(users_df.add_suffix('_user'), left_on='user_id', right_on='id_user', how='left')
            .merge(posts_df.add_suffix('_post'), left_on='post_id', right_on='id_post', how='left')
        )
        # Drop NaNs only for required columns later
        pass

        # Determine column names after suffixing
        is_viral_col = 'is_viral_post'
        if 'is_viral_post_post' in training_data.columns:
            is_viral_col = 'is_viral_post_post'
        elif 'is_viral_post' in training_data.columns:
            is_viral_col = 'is_viral_post'
        else:
            # fallback: create False
            training_data[is_viral_col] = False

        view_count_col = 'view_count_post' if 'view_count_post' in training_data.columns else 'view_count'
        if view_count_col not in training_data.columns:
            training_data[view_count_col] = 0

        # Final simplified frame
        final_df = training_data[[
            'user_id', 'post_id', 'label', 'user_persona',
            'post_is_trending', is_viral_col, view_count_col
        ]].copy()

        final_df.rename(columns={is_viral_col: 'is_viral_post', view_count_col: 'view_count_post'}, inplace=True)

        final_df['post_is_trending'] = final_df['post_is_trending'].astype(int)
        final_df['is_viral_post'] = final_df['is_viral_post'].astype(int)

        return final_df
    def generate_full_dataset(self, n_users=10, n_posts=20, n_interactions=100, save_path='data/'):
        print("🚀 Bắt đầu tạo dataset cho CTU Connect (phiên bản ĐẦY ĐỦ)...")
        os.makedirs(save_path, exist_ok=True)

        trending_tags = random.sample(self.potential_trending_tags, k=random.randint(1, 2))
        print(f"🔥 Chủ đề nóng lần này: {trending_tags}")

        print("👥 Tạo dữ liệu users với personas...")
        users_df = self.generate_users(n_users)

        print("📄 Tạo dữ liệu posts (trend/viral + nội dung động)...")
        posts_df = self.generate_posts(users_df, n_posts, trending_tags)
        # Đổi tên cột để merge rõ ràng
        posts_df = posts_df.rename(columns={'is_viral': 'is_viral_post'})

        print("🤝 Tạo dữ liệu interactions (có HIDE_POST)...")
        interactions_df = self.generate_interactions(users_df, posts_df, n_interactions)

        print("💬 Tạo dữ liệu comments (với nội dung logic dựa trên chất lượng post)...")
        comments_df, posts_df = self.generate_comments(users_df, posts_df, interactions_df)

        print("📊 Tạo training dataset [-1,0,1]...")
        training_df = self.create_training_dataset(users_df, posts_df, interactions_df)

        # Lưu file
        users_df.to_csv(f'{save_path}/ctu_connect_users.csv', index=False)
        posts_df.to_csv(f'{save_path}/ctu_connect_posts.csv', index=False)
        interactions_df.to_csv(f'{save_path}/ctu_connect_interactions.csv', index=False)
        comments_df.to_csv(f'{save_path}/ctu_connect_comments.csv', index=False)
        training_df.to_csv(f'{save_path}/ctu_connect_training.csv', index=False)

        print("\n✅ Dataset ĐẦY ĐỦ đã được tạo thành công!")
        print(f"📁 Vị trí: {save_path}")
        print(f"👥 Users: {len(users_df)} | 📄 Posts: {len(posts_df)} | 🤝 Interactions: {len(interactions_df)} | 💬 Comments: {len(comments_df)}")
        print(f"📊 Phân bố nhãn trong training set: {Counter(training_df['label'])}")

        return {'users': users_df, 'posts': posts_df, 'interactions': interactions_df, 'comments': comments_df, 'training': training_df}

if __name__ == "__main__":
    generator = CTUConnectDataGenerator(seed=42)
    dataset = generator.generate_full_dataset(
        n_users=10000,
        n_posts=1000000,
        n_interactions=1000000,
        save_path='/content/drive/MyDrive/ctu_connect/dataset'
    )