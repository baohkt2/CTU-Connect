package com.ctuconnect.controller;

import com.ctuconnect.dto.*;
import com.ctuconnect.service.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/users/categories")
@RequiredArgsConstructor
public class CategoriesController {
    private final BatchService batchService;
    private final CollegeService collegeService;
    private final FacultyService facultyService;
    private final MajorService majorService;
    private final GenderService genderService;

    @GetMapping("/all")
    public ResponseEntity<Map<String, Object>> getAllCategories() {
        Map<String, Object> categories = new HashMap<>();

        // Get all colleges with their faculties and majors nested
        List<CollegeDTO> colleges = collegeService.getAllColleges();
        List<CollegeWithHierarchyDTO> collegesWithHierarchy = colleges.stream()
                .map(college -> {
                    List<FacultyDTO> facultiesInCollege = facultyService.getFacultiesByCollege(college.getName());
                    List<FacultyWithMajorsDTO> facultiesWithMajors = facultiesInCollege.stream()
                            .map(faculty -> {
                                List<MajorDTO> majorsInFaculty = majorService.getMajorsByFaculty(faculty.getName());
                                return FacultyWithMajorsDTO.builder()
                                        .name(faculty.getName())
                                        .code(faculty.getCode())
                                        .collegeName(faculty.getCollegeName())
                                        .majors(majorsInFaculty)
                                        .build();
                            })
                            .collect(Collectors.toList());

                    return CollegeWithHierarchyDTO.builder()
                            .name(college.getName())
                            .code(college.getCode())
                            .faculties(facultiesWithMajors)
                            .build();
                })
                .collect(Collectors.toList());

        categories.put("colleges", collegesWithHierarchy);
        categories.put("batches", batchService.getAllBatches());
        categories.put("genders", genderService.getAllGenders());

        return ResponseEntity.ok(categories);
    }

    @GetMapping("/hierarchy")
    public ResponseEntity<Map<String, Object>> getCategoriesHierarchy() {
        Map<String, Object> hierarchy = new HashMap<>();
        List<CollegeDTO> colleges = collegeService.getAllColleges();

        Map<String, Object> collegeHierarchy = new HashMap<>();
        for (CollegeDTO college : colleges) {
            List<FacultyDTO> faculties = facultyService.getFacultiesByCollege(college.getName());
            Map<String, Object> facultyHierarchy = new HashMap<>();

            for (FacultyDTO faculty : faculties) {
                List<MajorDTO> majors = majorService.getMajorsByFaculty(faculty.getName());
                facultyHierarchy.put(faculty.getName(), Map.of(
                    "faculty", faculty,
                    "majors", majors
                ));
            }

            collegeHierarchy.put(college.getName(), Map.of(
                "college", college,
                "faculties", facultyHierarchy
            ));
        }

        hierarchy.put("colleges", collegeHierarchy);
        hierarchy.put("batches", batchService.getAllBatches());
        hierarchy.put("genders", genderService.getAllGenders());

        return ResponseEntity.ok(hierarchy);
    }
}
