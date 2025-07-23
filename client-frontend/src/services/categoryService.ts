import api from '@/lib/api';
import { BatchInfo, CollegeInfo, FacultyInfo, MajorInfo, GenderInfo, HierarchicalCategories } from '@/types';

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

  async getCollegeByName(name: string): Promise<CollegeInfo> {
    const response = await api.get(`/users/colleges/${encodeURIComponent(name)}`);
    return response.data;
  },

  // Faculty APIs
  async getFaculties(): Promise<FacultyInfo[]> {
    const response = await api.get('/users/faculties');
    return response.data;
  },

  async getFacultiesByCollege(collegeName: string): Promise<FacultyInfo[]> {
    const response = await api.get(`/users/faculties/college/${encodeURIComponent(collegeName)}`);
    return response.data;
  },

  async getFacultyByName(name: string): Promise<FacultyInfo> {
    const response = await api.get(`/users/faculties/${encodeURIComponent(name)}`);
    return response.data;
  },

  // Major APIs
  async getMajors(): Promise<MajorInfo[]> {
    const response = await api.get('/users/majors');
    return response.data;
  },

  async getMajorsByFaculty(facultyName: string): Promise<MajorInfo[]> {
    const response = await api.get(`/users/majors/faculty/${encodeURIComponent(facultyName)}`);
    return response.data;
  },

  async getMajorByName(name: string): Promise<MajorInfo> {
    const response = await api.get(`/users/majors/${encodeURIComponent(name)}`);
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
  async getAllCategories(): Promise<HierarchicalCategories> {
    const response = await api.get('/users/categories/all');
    return response.data;
  },

  // Get categories in hierarchy
  async getCategoriesHierarchy(): Promise<{
    colleges: { [key: string]: { college: CollegeInfo; faculties: { [key: string]: { faculty: FacultyInfo; majors: MajorInfo[] } } } };
    batches: BatchInfo[];
    genders: GenderInfo[];
  }> {
    const response = await api.get('/users/categories/hierarchy');
    return response.data;
  },

  // Helper methods to extract flat lists from hierarchical data
  async getFlatColleges(): Promise<CollegeInfo[]> {
    const hierarchical = await this.getAllCategories();
    return hierarchical.colleges.map(college => ({
      name: college.name,
      code: college.code
    }));
  },

  async getFlatFaculties(): Promise<FacultyInfo[]> {
    const hierarchical = await this.getAllCategories();
    const faculties: FacultyInfo[] = [];

    hierarchical.colleges.forEach(college => {
      college.faculties.forEach(faculty => {
        faculties.push({
          name: faculty.name,
          code: faculty.code,
          college: { name: college.name, code: college.code }
        });
      });
    });

    return faculties;
  },

  async getFlatMajors(): Promise<MajorInfo[]> {
    const hierarchical = await this.getAllCategories();
    const majors: MajorInfo[] = [];

    hierarchical.colleges.forEach(college => {
      college.faculties.forEach(faculty => {
        faculty.majors.forEach(major => {
          majors.push({
            name: major.name,
            code: major.code,
            faculty: {
              name: faculty.name,
              code: faculty.code,
              college: { name: college.name, code: college.code }
            }
          });
        });
      });
    });

    return majors;
  },

  // Get faculties for a specific college from hierarchical data
  async getFacultiesFromHierarchy(collegeName: string): Promise<FacultyInfo[]> {
    const hierarchical = await this.getAllCategories();
    const college = hierarchical.colleges.find(c => c.name === collegeName);

    if (!college) return [];

    return college.faculties.map(faculty => ({
      name: faculty.name,
      code: faculty.code,
      college: { name: college.name, code: college.code }
    }));
  },

  // Get majors for a specific faculty from hierarchical data
  async getMajorsFromHierarchy(facultyName: string): Promise<MajorInfo[]> {
    const hierarchical = await this.getAllCategories();

    for (const college of hierarchical.colleges) {
      const faculty = college.faculties.find(f => f.name === facultyName);
      if (faculty) {
        return faculty.majors.map(major => ({
          name: major.name,
          code: major.code,
          faculty: {
            name: faculty.name,
            code: faculty.code,
            college: { name: college.name, code: college.code }
          }
        }));
      }
    }

    return [];
  },
};
