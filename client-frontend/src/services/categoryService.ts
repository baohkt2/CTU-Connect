import api from '@/lib/api';
import { BatchInfo, CollegeInfo, FacultyInfo, MajorInfo, GenderInfo } from '@/types';

export const categoryService = {
  // Batch APIs
  async getBatches(): Promise<BatchInfo[]> {
    const response = await api.get('/users/batches');
    return response.data;
  },

  async getBatchByYear(year: number): Promise<BatchInfo> {
    const response = await api.get(`/users/batches/${year}`);
    return response.data;
  },

  // College APIs
  async getColleges(): Promise<CollegeInfo[]> {
    const response = await api.get('/users/colleges');
    return response.data;
  },

  async getCollegeByCode(code: string): Promise<CollegeInfo> {
    const response = await api.get(`/users/colleges/${code}`);
    return response.data;
  },

  // Faculty APIs
  async getFaculties(): Promise<FacultyInfo[]> {
    const response = await api.get('/users/faculties');
    return response.data;
  },

  async getFacultiesByCollege(collegeCode: string): Promise<FacultyInfo[]> {
    const response = await api.get(`/users/faculties/college/${collegeCode}`);
    return response.data;
  },

  async getFacultyByCode(code: string): Promise<FacultyInfo> {
    const response = await api.get(`/users/faculties/${code}`);
    return response.data;
  },

  // Major APIs
  async getMajors(): Promise<MajorInfo[]> {
    const response = await api.get('/users/majors');
    return response.data;
  },

  async getMajorsByFaculty(facultyCode: string): Promise<MajorInfo[]> {
    const response = await api.get(`/users/majors/faculty/${facultyCode}`);
    return response.data;
  },

  async getMajorByCode(code: string): Promise<MajorInfo> {
    const response = await api.get(`/users/majors/${code}`);
    return response.data;
  },

  // Gender APIs
  async getGenders(): Promise<GenderInfo[]> {
    const response = await api.get('/users/genders');
    return response.data;
  },

  async getGenderByCode(code: string): Promise<GenderInfo> {
    const response = await api.get(`/users/genders/${code}`);
    return response.data;
  },

  // Combined APIs for performance
  async getAllCategories(): Promise<{
    batches: BatchInfo[];
    colleges: CollegeInfo[];
    faculties: FacultyInfo[];
    majors: MajorInfo[];
    genders: GenderInfo[];
  }> {
    const response = await api.get('/users/categories/all');
    return response.data;
  },

  // Get categories in hierarchy (college -> faculty -> major)
  async getCategoriesHierarchy(): Promise<{
    colleges: { [key: string]: { college: CollegeInfo; faculties: { [key: string]: { faculty: FacultyInfo; majors: MajorInfo[] } } } };
    batches: BatchInfo[];
    genders: GenderInfo[];
  }> {
    const response = await api.get('/users/categories/hierarchy');
    return response.data;
  },
};
