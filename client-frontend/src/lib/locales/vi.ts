// Vietnamese localization constants
export const VI_LOCALE = {
  // Common actions
  actions: {
    save: 'Lưu',
    cancel: 'Hủy',
    delete: 'Xóa',
    edit: 'Chỉnh sửa',
    submit: 'Gửi',
    confirm: 'Xác nhận',
    back: 'Quay lại',
    next: 'Tiếp theo',
    loading: 'Đang tải...',
    retry: 'Thử lại',
    close: 'Đóng',
    open: 'Mở',
    view: 'Xem',
    share: 'Chia sẻ',
    copy: 'Sao chép',
    download: 'Tải xuống',
    upload: 'Tải lên',
    search: 'Tìm kiếm',
    filter: 'Lọc',
    sort: 'Sắp xếp',
    refresh: 'Làm mới'
  },

  // Authentication
  auth: {
    login: 'Đăng nhập',
    logout: 'Đăng xuất',
    register: 'Đăng ký',
    forgotPassword: 'Quên mật khẩu',
    resetPassword: 'Đặt lại mật khẩu',
    changePassword: 'Đổi mật khẩu',
    verifyEmail: 'Xác thực email',
    resendVerification: 'Gửi lại mã xác thực'
  },

  // Posts
  posts: {
    createPost: 'Tạo bài viết',
    editPost: 'Chỉnh sửa bài viết',
    deletePost: 'Xóa bài viết',
    sharePost: 'Chia sẻ bài viết',
    likePost: 'Thích bài viết',
    unlikePost: 'Bỏ thích',
    bookmarkPost: 'Lưu bài viết',
    removeBookmark: 'Bỏ lưu',
    commentPost: 'Bình luận',
    viewComments: 'Xem bình luận',
    hideComments: 'Ẩn bình luận',
    writeComment: 'Viết bình luận...',
    replyComment: 'Trả lời',
    postTitle: 'Tiêu đề bài viết',
    postContent: 'Nội dung bài viết',
    addMedia: 'Thêm ảnh/video',
    addTag: 'Thêm thẻ',
    selectCategory: 'Chọn danh mục',
    noPostsFound: 'Không tìm thấy bài viết nào',
    loadMorePosts: 'Tải thêm bài viết'
  },

  // User profile
  profile: {
    profile: 'Hồ sơ',
    editProfile: 'Chỉnh sửa hồ sơ',
    viewProfile: 'Xem hồ sơ',
    fullName: 'Họ và tên',
    email: 'Email',
    username: 'Tên đăng nhập',
    bio: 'Giới thiệu',
    avatar: 'Ảnh đại diện',
    coverPhoto: 'Ảnh bìa',
    birthday: 'Ngày sinh',
    gender: 'Giới tính',
    faculty: 'Khoa',
    major: 'Ngành học',
    studentId: 'Mã số sinh viên',
    academicYear: 'Niên khóa',
    phone: 'Số điện thoại',
    address: 'Địa chỉ'
  },

  // Messages and notifications
  messages: {
    success: {
      loginSuccess: 'Đăng nhập thành công',
      logoutSuccess: 'Đăng xuất thành công',
      registrationSuccess: 'Đăng ký thành công',
      profileUpdated: 'Cập nhật hồ sơ thành công',
      postCreated: 'Tạo bài viết thành công',
      postUpdated: 'Cập nhật bài viết thành công',
      postDeleted: 'Xóa bài viết thành công',
      commentAdded: 'Thêm bình luận thành công',
      passwordChanged: 'Đổi mật khẩu thành công',
      emailVerified: 'Xác thực email thành công'
    },
    error: {
      genericError: 'Đã xảy ra lỗi. Vui lòng thử lại.',
      networkError: 'Lỗi kết nối mạng. Vui lòng kiểm tra kết nối internet.',
      unauthorized: 'Bạn không có quyền truy cập.',
      forbidden: 'Truy cập bị từ chối.',
      notFound: 'Không tìm thấy tài nguyên.',
      serverError: 'Lỗi máy chủ. Vui lòng thử lại sau.',
      validationError: 'Dữ liệu không hợp lệ.',
      loginFailed: 'Đăng nhập thất bại',
      registrationFailed: 'Đăng ký thất bại',
      uploadFailed: 'Tải file thất bại',
      securityError: 'Xác thực bảo mật thất bại. Vui lòng thử lại.'
    },
    validation: {
      required: 'Trường này là bắt buộc',
      invalidEmail: 'Email không hợp lệ',
      invalidPassword: 'Mật khẩu không hợp lệ',
      passwordMismatch: 'Mật khẩu xác nhận không khớp',
      minLength: 'Tối thiểu {min} ký tự',
      maxLength: 'Tối đa {max} ký tự',
      invalidFormat: 'Định dạng không hợp lệ'
    }
  },

  // Time formatting
  time: {
    now: 'Vừa xong',
    minutesAgo: '{count} phút trước',
    hoursAgo: '{count} giờ trước',
    daysAgo: '{count} ngày trước',
    weeksAgo: '{count} tuần trước',
    monthsAgo: '{count} tháng trước',
    yearsAgo: '{count} năm trước'
  },

  // File upload
  upload: {
    selectFiles: 'Chọn tệp',
    dragAndDrop: 'Kéo thả tệp vào đây',
    maxFileSize: 'Kích thước tệp tối đa: {size}MB',
    supportedFormats: 'Định dạng hỗ trợ: {formats}',
    uploadProgress: 'Đang tải lên... {progress}%',
    uploadComplete: 'Tải lên hoàn tất',
    uploadFailed: 'Tải lên thất bại'
  },

  // Search and filters
  search: {
    searchPlaceholder: 'Tìm kiếm...',
    noResults: 'Không tìm thấy kết quả',
    searchResults: 'Kết quả tìm kiếm',
    filters: 'Bộ lọc',
    sortBy: 'Sắp xếp theo',
    sortNewest: 'Mới nhất',
    sortOldest: 'Cũ nhất',
    sortMostLiked: 'Nhiều lượt thích nhất',
    sortMostViewed: 'Nhiều lượt xem nhất'
  },

  // Categories
  categories: {
    academic: 'Học tập',
    social: 'Sinh hoạt',
    announcement: 'Thông báo',
    career: 'Nghề nghiệp',
    technology: 'Công nghệ',
    sports: 'Thể thao',
    entertainment: 'Giải trí',
    other: 'Khác'
  }
} as const;

export type LocaleKey = keyof typeof VI_LOCALE;
export type LocaleValue = typeof VI_LOCALE[LocaleKey];
