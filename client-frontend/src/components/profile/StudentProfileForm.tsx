'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { userService } from '@/services/userService';
import { categoryService } from '@/services/categoryService';
import { User, StudentProfileUpdateRequest, MajorInfo, FacultyInfo, GenderInfo, BatchInfo, CollegeInfo } from '@/types';
import { useToast } from '@/hooks/useToast';

interface StudentProfileFormProps {
  user: User;
}

export default function StudentProfileForm({ user }: StudentProfileFormProps) {
  const router = useRouter();
  const { showToast } = useToast();

  const [formData, setFormData] = useState<StudentProfileUpdateRequest>({
    fullName: user.fullName || '',
    bio: user.bio || '',
    studentId: user.studentId || '',
    majorCode: user.major?.code || '',
    batchYear: user.batch?.year || new Date().getFullYear(),
    genderCode: user.gender?.code || '',
    avatarUrl: user.avatarUrl || '',
    backgroundUrl: user.backgroundUrl || ''
  });

  const [dropdownData, setDropdownData] = useState({
    majors: [] as MajorInfo[],
    faculties: [] as FacultyInfo[],
    colleges: [] as CollegeInfo[],
    genders: [] as GenderInfo[],
    batches: [] as BatchInfo[]
  });

  const [filteredFaculties, setFilteredFaculties] = useState<FacultyInfo[]>([]);
  const [filteredMajors, setFilteredMajors] = useState<MajorInfo[]>([]);
  const [selectedCollege, setSelectedCollege] = useState<string>('');
  const [selectedFaculty, setSelectedFaculty] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [dataLoading, setDataLoading] = useState(true);

  useEffect(() => {
    const loadDropdownData = async () => {
      try {
        // Use the new getAllCategories API for better performance
        const categories = await categoryService.getAllCategories();

        console.log(categories);

        setDropdownData({
          majors: categories.majors,
          faculties: categories.faculties,
          colleges: categories.colleges,
          genders: categories.genders,
          batches: categories.batches
        });

        // Set initial filtered data
        setFilteredFaculties(categories.faculties);
        setFilteredMajors(categories.majors);

        // If user has existing data, set the selected values
        if (user.major?.faculty?.college?.code) {
          setSelectedCollege(user.major.faculty.college.code);
          const facultiesInCollege = categories.faculties.filter(f => f.college?.code === user.major?.faculty?.college?.code);
          setFilteredFaculties(facultiesInCollege);
        }

        if (user.major?.faculty?.code) {
          setSelectedFaculty(user.major.faculty.code);
          const majorsInFaculty = categories.majors.filter(m => m.faculty?.code === user.major?.faculty?.code);
          setFilteredMajors(majorsInFaculty);
        }
      } catch (error) {
        console.error('Error loading dropdown data:', error);
        showToast('Không thể tải dữ liệu danh mục', 'error');
      } finally {
        setDataLoading(false);
      }
    };

    loadDropdownData();
  }, [user.major, showToast]);

  const handleCollegeChange = async (collegeCode: string) => {
    setSelectedCollege(collegeCode);
    setSelectedFaculty('');
    setFormData({ ...formData, majorCode: '' }); // Reset major and faculty when college changes

    if (collegeCode) {
      try {
        const faculties = await categoryService.getFacultiesByCollege(collegeCode);
        setFilteredFaculties(faculties);
        setFilteredMajors([]); // Clear majors when college changes
      } catch (error) {
        console.error('Error loading faculties for college:', error);
        showToast('Không thể tải danh sách khoa', 'error');
      }
    } else {
      setFilteredFaculties(dropdownData.faculties);
      setFilteredMajors(dropdownData.majors);
    }
  };

  const handleFacultyChange = async (facultyCode: string) => {
    setSelectedFaculty(facultyCode);
    setFormData({ ...formData, majorCode: '' }); // Reset major when faculty changes

    if (facultyCode) {
      try {
        const majors = await categoryService.getMajorsByFaculty(facultyCode);
        setFilteredMajors(majors);
      } catch (error) {
        console.error('Error loading majors for faculty:', error);
        showToast('Không thể tải danh sách ngành', 'error');
      }
    } else {
      setFilteredMajors(dropdownData.majors);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.fullName || !formData.studentId || !formData.majorCode || !formData.genderCode) {
      showToast('Vui lòng điền đầy đủ thông tin bắt buộc', 'error');
      return;
    }

    setLoading(true);
    try {
      await userService.updateMyProfile(formData);
      showToast('Cập nhật thông tin thành công!', 'success');
      router.push('/'); // Redirect to home page after successful update
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
    <form onSubmit={handleSubmit} className="space-y-6 text-gray-700">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Full Name */}
        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
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

        {/* Student ID */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Mã số sinh viên <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            value={formData.studentId}
            onChange={(e) => setFormData({ ...formData, studentId: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
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

        {/* Faculty */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Khoa <span className="text-red-500">*</span>
          </label>
          <select
            value={selectedFaculty}
            onChange={(e) => handleFacultyChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">Chọn khoa</option>
            {filteredFaculties.map((faculty) => (
              <option key={faculty.code} value={faculty.code}>
                {faculty.name}
              </option>
            ))}
          </select>
        </div>

        {/* Major */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Ngành học <span className="text-red-500">*</span>
          </label>
          <select
            value={formData.majorCode}
            onChange={(e) => setFormData({ ...formData, majorCode: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
            disabled={!selectedFaculty}
          >
            <option value="">Chọn ngành học</option>
            {filteredMajors.map((major) => (
              <option key={major.code} value={major.code}>
                {major.name}
              </option>
            ))}
          </select>
        </div>

        {/* Batch Year */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Niên khóa <span className="text-red-500">*</span>
          </label>
          <select
            value={formData.batchYear}
            onChange={(e) => setFormData({ ...formData, batchYear: parseInt(e.target.value) })}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            {dropdownData.batches.map((batch) => (
              <option key={batch.year} value={batch.year}>
                {batch.year}
              </option>
            ))}
          </select>
        </div>

        {/* Gender */}
        <div>
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
          placeholder="Viết vài dòng giới thiệu về bản thân..."
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
