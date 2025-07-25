package com.ctuconnect.service;

import com.ctuconnect.dto.CategoryDTO;
import com.ctuconnect.entity.*;
import com.ctuconnect.repository.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class CategoryService {

    @Autowired
    private CollegeRepository collegeRepository;

    @Autowired
    private FacultyRepository facultyRepository;

    @Autowired
    private MajorRepository majorRepository;

    @Autowired
    private BatchRepository batchRepository;

    @Autowired
    private GenderRepository genderRepository;

    // ===================== HIERARCHICAL DATA RETRIEVAL =====================

    public CategoryDTO.HierarchicalCategories getAllCategoriesHierarchical() {
        // Get all colleges
        List<CollegeEntity> colleges = collegeRepository.findAll();
        List<BatchEntity> batches = batchRepository.findAllByOrderByYearDesc();
        List<GenderEntity> genders = genderRepository.findAll();

        // Build hierarchical structure manually
        List<CategoryDTO.CollegeInfo> collegeInfos = colleges.stream()
                .map(college -> {
                    // Get faculties for this college
                    List<FacultyEntity> faculties = facultyRepository.findByCollege(college.getName());

                    List<CategoryDTO.FacultyInfo> facultyInfos = faculties.stream()
                            .map(faculty -> {
                                // Get majors for this faculty
                                List<MajorEntity> majors = majorRepository.findByFaculty(faculty.getName());

                                List<CategoryDTO.MajorInfo> majorInfos = majors.stream()
                                        .map(major -> CategoryDTO.MajorInfo.builder()
                                                .name(major.getName())
                                                .code(major.getCode())
                                                .build())
                                        .collect(Collectors.toList());

                                return CategoryDTO.FacultyInfo.builder()
                                        .name(faculty.getName())
                                        .code(faculty.getCode())
                                        .majors(majorInfos)
                                        .build();
                            })
                            .collect(Collectors.toList());

                    return CategoryDTO.CollegeInfo.builder()
                            .name(college.getName())
                            .code(college.getCode())
                            .faculties(facultyInfos)
                            .build();
                })
                .collect(Collectors.toList());

        List<CategoryDTO.BatchInfo> batchInfos = batches.stream()
                .map(this::convertToBatchInfo)
                .collect(Collectors.toList());

        List<CategoryDTO.GenderInfo> genderInfos = genders.stream()
                .map(this::convertToGenderInfo)
                .collect(Collectors.toList());

        return CategoryDTO.HierarchicalCategories.builder()
                .colleges(collegeInfos)
                .batches(batchInfos)
                .genders(genderInfos)
                .build();
    }

    // ===================== COLLEGE OPERATIONS =====================

    public List<CategoryDTO.CollegeInfo> getAllColleges() {
        List<CollegeEntity> colleges = collegeRepository.findAll();
        return colleges.stream()
                .map(college -> {
                    // Get faculties for this college
                    List<FacultyEntity> faculties = facultyRepository.findByCollege(college.getName());

                    List<CategoryDTO.FacultyInfo> facultyInfos = faculties.stream()
                            .map(faculty -> {
                                // Get majors for this faculty
                                List<MajorEntity> majors = majorRepository.findByFaculty(faculty.getName());

                                List<CategoryDTO.MajorInfo> majorInfos = majors.stream()
                                        .map(major -> CategoryDTO.MajorInfo.builder()
                                                .name(major.getName())
                                                .code(major.getCode())
                                                .build())
                                        .collect(Collectors.toList());

                                return CategoryDTO.FacultyInfo.builder()
                                        .name(faculty.getName())
                                        .code(faculty.getCode())
                                        .majors(majorInfos)
                                        .build();
                            })
                            .collect(Collectors.toList());

                    return CategoryDTO.CollegeInfo.builder()
                            .name(college.getName())
                            .code(college.getCode())
                            .faculties(facultyInfos)
                            .build();
                })
                .collect(Collectors.toList());
    }

    public CategoryDTO.CollegeInfo getCollegeByName(String name) {
        CollegeEntity college = collegeRepository.findByName(name)
                .orElseThrow(() -> new RuntimeException("College not found: " + name));

        // Get faculties for this college
        List<FacultyEntity> faculties = facultyRepository.findByCollege(college.getName());

        List<CategoryDTO.FacultyInfo> facultyInfos = faculties.stream()
                .map(faculty -> {
                    // Get majors for this faculty
                    List<MajorEntity> majors = majorRepository.findByFaculty(faculty.getName());

                    List<CategoryDTO.MajorInfo> majorInfos = majors.stream()
                            .map(major -> CategoryDTO.MajorInfo.builder()
                                    .name(major.getName())
                                    .code(major.getCode())
                                    .build())
                            .collect(Collectors.toList());

                    return CategoryDTO.FacultyInfo.builder()
                            .name(faculty.getName())
                            .code(faculty.getCode())
                            .majors(majorInfos)
                            .build();
                })
                .collect(Collectors.toList());

        return CategoryDTO.CollegeInfo.builder()
                .name(college.getName())
                .code(college.getCode())
                .faculties(facultyInfos)
                .build();
    }

    // ===================== FACULTY OPERATIONS =====================

    public List<CategoryDTO.FacultyInfo> getAllFaculties() {
        List<FacultyEntity> faculties = facultyRepository.findAll();
        return faculties.stream()
                .map(faculty -> {
                    // Get majors for this faculty
                    List<MajorEntity> majors = majorRepository.findByFaculty(faculty.getName());

                    List<CategoryDTO.MajorInfo> majorInfos = majors.stream()
                            .map(major -> CategoryDTO.MajorInfo.builder()
                                    .name(major.getName())
                                    .code(major.getCode())
                                    .build())
                            .collect(Collectors.toList());

                    CategoryDTO.CollegeBasicInfo college = CategoryDTO.CollegeBasicInfo.builder()
                            .name(faculty.getCollege())
                            .build();

                    return CategoryDTO.FacultyInfo.builder()
                            .name(faculty.getName())
                            .code(faculty.getCode())
                            .college(college)
                            .majors(majorInfos)
                            .build();
                })
                .collect(Collectors.toList());
    }

    public List<CategoryDTO.FacultyInfo> getFacultiesByCollege(String collegeName) {
        List<FacultyEntity> faculties = facultyRepository.findByCollege(collegeName);
        return faculties.stream()
                .map(faculty -> {
                    // Get majors for this faculty
                    List<MajorEntity> majors = majorRepository.findByFaculty(faculty.getName());

                    List<CategoryDTO.MajorInfo> majorInfos = majors.stream()
                            .map(major -> CategoryDTO.MajorInfo.builder()
                                    .name(major.getName())
                                    .code(major.getCode())
                                    .build())
                            .collect(Collectors.toList());

                    CategoryDTO.CollegeBasicInfo college = CategoryDTO.CollegeBasicInfo.builder()
                            .name(faculty.getCollege())
                            .build();

                    return CategoryDTO.FacultyInfo.builder()
                            .name(faculty.getName())
                            .code(faculty.getCode())
                            .college(college)
                            .majors(majorInfos)
                            .build();
                })
                .collect(Collectors.toList());
    }

    public CategoryDTO.FacultyInfo getFacultyByName(String name) {
        FacultyEntity faculty = facultyRepository.findByName(name)
                .orElseThrow(() -> new RuntimeException("Faculty not found: " + name));

        // Get majors for this faculty
        List<MajorEntity> majors = majorRepository.findByFaculty(faculty.getName());

        List<CategoryDTO.MajorInfo> majorInfos = majors.stream()
                .map(major -> CategoryDTO.MajorInfo.builder()
                        .name(major.getName())
                        .code(major.getCode())
                        .build())
                .collect(Collectors.toList());

        CategoryDTO.CollegeBasicInfo college = CategoryDTO.CollegeBasicInfo.builder()
                .name(faculty.getCollege())
                .build();

        return CategoryDTO.FacultyInfo.builder()
                .name(faculty.getName())
                .code(faculty.getCode())
                .college(college)
                .majors(majorInfos)
                .build();
    }

    // ===================== MAJOR OPERATIONS =====================

    public List<CategoryDTO.MajorInfo> getAllMajors() {
        List<MajorEntity> majors = majorRepository.findAll();
        return majors.stream()
                .map(major -> {
                    CategoryDTO.FacultyBasicInfo faculty = CategoryDTO.FacultyBasicInfo.builder()
                            .name(major.getFaculty())
                            .build();

                    return CategoryDTO.MajorInfo.builder()
                            .name(major.getName())
                            .code(major.getCode())
                            .faculty(faculty)
                            .build();
                })
                .collect(Collectors.toList());
    }

    public List<CategoryDTO.MajorInfo> getMajorsByFaculty(String facultyName) {
        List<MajorEntity> majors = majorRepository.findByFaculty(facultyName);
        return majors.stream()
                .map(major -> {
                    CategoryDTO.FacultyBasicInfo faculty = CategoryDTO.FacultyBasicInfo.builder()
                            .name(major.getFaculty())
                            .build();

                    return CategoryDTO.MajorInfo.builder()
                            .name(major.getName())
                            .code(major.getCode())
                            .faculty(faculty)
                            .build();
                })
                .collect(Collectors.toList());
    }

    public CategoryDTO.MajorInfo getMajorByName(String name) {
        MajorEntity major = majorRepository.findByName(name)
                .orElseThrow(() -> new RuntimeException("Major not found: " + name));

        CategoryDTO.FacultyBasicInfo faculty = CategoryDTO.FacultyBasicInfo.builder()
                .name(major.getFaculty())
                .build();

        return CategoryDTO.MajorInfo.builder()
                .name(major.getName())
                .code(major.getCode())
                .faculty(faculty)
                .build();
    }

    // ===================== BATCH OPERATIONS =====================

    public List<CategoryDTO.BatchInfo> getAllBatches() {
        List<BatchEntity> batches = batchRepository.findAllByOrderByYearDesc();
        return batches.stream()
                .map(this::convertToBatchInfo)
                .collect(Collectors.toList());
    }

    public CategoryDTO.BatchInfo getBatchByYear(Integer year) {
        BatchEntity batch = batchRepository.findByYear(year)
                .orElseThrow(() -> new RuntimeException("Batch not found: " + year));
        return convertToBatchInfo(batch);
    }

    // ===================== GENDER OPERATIONS =====================

    public List<CategoryDTO.GenderInfo> getAllGenders() {
        List<GenderEntity> genders = genderRepository.findAll();
        return genders.stream()
                .map(this::convertToGenderInfo)
                .collect(Collectors.toList());
    }

    public CategoryDTO.GenderInfo getGenderByCode(String code) {
        GenderEntity gender = genderRepository.findByCode(code)
                .orElseThrow(() -> new RuntimeException("Gender not found: " + code));
        return convertToGenderInfo(gender);
    }

    // ===================== CONVERSION METHODS =====================

    private CategoryDTO.BatchInfo convertToBatchInfo(BatchEntity batch) {
        return CategoryDTO.BatchInfo.builder()
                .year(batch.getYear())
                .build();
    }

    private CategoryDTO.GenderInfo convertToGenderInfo(GenderEntity gender) {
        return CategoryDTO.GenderInfo.builder()
                .code(gender.getCode())
                .name(gender.getName())
                .build();
    }
}
