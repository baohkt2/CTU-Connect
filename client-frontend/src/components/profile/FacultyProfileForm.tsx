'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { userService } from '@/services/userService';
import { categoryService } from '@/services/categoryService';
import { User, FacultyProfileUpdateRequest, FacultyInfo, GenderInfo, CollegeInfo, HierarchicalCategories } from '@/types';
import { useToast } from '@/hooks/useToast';
import { useAuth } from '@/contexts/AuthContext';

interface FacultyProfileFormProps {
  user: User;
}

export default function FacultyProfileForm({ user }: FacultyProfileFormProps) {
  const router = useRouter();
  const { showToast } = useToast();
  const { updateUser } = useAuth();

  const [formData, setFormData] = useState<FacultyProfileUpdateRequest>({
    fullName: user.fullName || '',
    bio: user.bio || '',
    staffCode: user.staffCode || '',
    position: user.position || '',
    academicTitle: user.academicTitle || '',
    degree: user.degree || '',
    workingFacultyName: user.workingFaculty?.name || '', // Đổi từ workingFacultyCode sang workingFacultyName
    genderCode: user.gender?.code || '',
    avatarUrl: user.avatarUrl || '',
    backgroundUrl: user.backgroundUrl || ''
  });

  const [dropdownData, setDropdownData] = useState({
    hierarchicalData: null as HierarchicalCategories | null,
    colleges: [] as CollegeInfo[],
    faculties: [] as FacultyInfo[],
    genders: [] as GenderInfo[]
  });

  const [filteredFaculties, setFilteredFaculties] = useState<FacultyInfo[]>([]);
  const [selectedCollege, setSelectedCollege] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [dataLoading, setDataLoading] = useState(true);

  // Position options for faculty
  const positionOptions = [
    { value: 'GIANG_VIEN', label: 'Giảng viên' },
    { value: 'GIANG_VIEN_CHINH', label: 'Giảng viên chính' },
    { value: 'PHO_GIAO_SU', label: 'Phó Giáo sư' },
    { value: 'GIAO_SU', label: 'Giáo sư' },
    { value: 'CAN_BO', label: 'Cán bộ' },
    { value: 'TRO_LY', label: 'Trợ lý' },
    { value: 'NGHIEN_CUU_VIEN', label: 'Nghiên cứu viên' }
  ];

  // Academic title options
  const academicTitleOptions = [
    { value: 'GIAO_SU', label: 'Giáo sư' },
    { value: 'PHO_GIAO_SU', label: 'Phó Giáo sư' },
    { value: 'TIEN_SI', label: 'Tiến sĩ' },
    { value: 'THAC_SI', label: 'Thạc sĩ' },
    { value: 'CU_NHAN', label: 'Cử nhân' }
  ];

  // Degree options
  const degreeOptions = [
    { value: 'TIEN_SI', label: 'Tiến sĩ' },
    { value: 'THAC_SI', label: 'Thạc sĩ' },
    { value: 'CU_NHAN', label: 'Cử nhân' },
    { value: 'KHAC', label: 'Khác' }
  ];

  useEffect(() => {
    const loadDropdownData = async () => {
      try {
        // Use the new hierarchical API for better performance
        const hierarchicalData = await categoryService.getAllCategories();

        // Extract flat lists for easier manipulation
        const colleges = hierarchicalData.colleges.map(college => ({
          code: college.code,
          name: college.name
        }));

        setDropdownData({
          hierarchicalData,
          colleges,
          faculties: [], // Will be populated when college is selected
          genders: hierarchicalData.genders
        });

        // If user has existing faculty data, set the selected values and populate dependent dropdowns
        if (user.workingFaculty?.college?.code) {
          const collegeCode = user.workingFaculty.college.code;
          setSelectedCollege(collegeCode);

          const selectedCollegeData = hierarchicalData.colleges.find(c => c.code === collegeCode);
          if (selectedCollegeData) {
            const facultiesInCollege = selectedCollegeData.faculties.map(faculty => ({
              code: faculty.code,
              name: faculty.name,
              college: { code: collegeCode, name: selectedCollegeData.name }
            }));
            setFilteredFaculties(facultiesInCollege);
          }
        }
      } catch (error) {
        console.error('Error loading dropdown data:', error);
        showToast('Không thể tải dữ liệu danh mục', 'error');
      } finally {
        setDataLoading(false);
      }
    };

    loadDropdownData();
  }, [user.workingFaculty, showToast]);

  const handleCollegeChange = (collegeCode: string) => {
    setSelectedCollege(collegeCode);
    setFormData({ ...formData, workingFacultyName: '' }); // Reset faculty when college changes

    if (collegeCode && dropdownData.hierarchicalData) {
      const selectedCollegeData = dropdownData.hierarchicalData.colleges.find(c => c.name === collegeCode);
      if (selectedCollegeData) {
        const facultiesInCollege = selectedCollegeData.faculties.map(faculty => ({
          name: faculty.name,
          code: faculty.code,
          college: { name: collegeCode, code: selectedCollegeData.code }
        }));
        setFilteredFaculties(facultiesInCollege);
      }
    } else {
      setFilteredFaculties([]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.fullName || !formData.staffCode || !formData.position || !formData.workingFacultyName || !formData.genderCode) {
      showToast('Vui lòng điền đầy đủ thông tin bắt buộc', 'error');
      return;
    }

    setLoading(true);
    try {
      const updatedUser = await userService.updateMyProfile(formData);

      // Update user context with new data
      updateUser(updatedUser);

      showToast('Cập nhật thông tin thành công!', 'success');

      // Redirect to home page immediately
      router.push('/');
    } catch (error) {
      console.error('Error updating profile:', error);
      showToast('Có lỗi xảy ra khi cập nhật thông tin', 'error');
    } finally {
      setLoading(false);
    }
  };

  if (dataLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Full Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Họ và tên <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            value={formData.fullName}
            onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        {/* Staff Code */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Mã cán bộ <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            value={formData.staffCode}
            onChange={(e) => setFormData({ ...formData, staffCode: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        {/* Position */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Chức vụ <span className="text-red-500">*</span>
          </label>
          <select
            value={formData.position}
            onChange={(e) => setFormData({ ...formData, position: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">Chọn chức vụ</option>
            {positionOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* College */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Trường <span className="text-red-500">*</span>
          </label>
          <select
            value={selectedCollege}
            onChange={(e) => handleCollegeChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">Chọn trường</option>
            {dropdownData.colleges.map((college) => (
              <option key={college.code} value={college.code}>
                {college.name}
              </option>
            ))}
          </select>
        </div>

        {/* Working Faculty */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Khoa làm việc <span className="text-red-500">*</span>
          </label>
          <select
            value={formData.workingFacultyName}
            onChange={(e) => setFormData({ ...formData, workingFacultyName: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">Chọn khoa</option>
            {filteredFaculties.map((faculty) => (
              <option key={faculty.name} value={faculty.name}>
                {faculty.name}
              </option>
            ))}
          </select>
        </div>

        {/* Academic Title */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Học hàm
          </label>
          <select
            value={formData.academicTitle}
            onChange={(e) => setFormData({ ...formData, academicTitle: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Chọn học hàm</option>
            {academicTitleOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Degree */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Học vị
          </label>
          <select
            value={formData.degree}
            onChange={(e) => setFormData({ ...formData, degree: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Chọn học vị</option>
            {degreeOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Gender */}
        <div className="md:col-span-1">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Giới tính <span className="text-red-500">*</span>
          </label>
          <select
            value={formData.genderCode}
            onChange={(e) => setFormData({ ...formData, genderCode: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">Chọn giới tính</option>
            {dropdownData.genders.map((gender) => (
              <option key={gender.code} value={gender.code}>
                {gender.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Bio */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Giới thiệu bản thân
        </label>
        <textarea
          value={formData.bio}
          onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
          rows={4}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Viết vài dòng giới thiệu về bản thân, kinh nghiệm làm việc..."
        />
      </div>

      {/* Avatar URL */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          URL ảnh đại diện
        </label>
        <input
          type="url"
          value={formData.avatarUrl}
          onChange={(e) => setFormData({ ...formData, avatarUrl: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="https://example.com/avatar.jpg"
        />
      </div>

      {/* Submit Button */}
      <div className="flex justify-end space-x-4">
        <button
          type="button"
          onClick={() => router.back()}
          className="px-6 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          Hủy
        </button>
        <button
          type="submit"
          disabled={loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          {loading ? 'Đang cập nhật...' : 'Cập nhật thông tin'}
        </button>
      </div>
    </form>
  );
}
