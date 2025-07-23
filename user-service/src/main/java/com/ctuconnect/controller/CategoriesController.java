package com.ctuconnect.controller;

import com.ctuconnect.dto.*;
import com.ctuconnect.service.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
        categories.put("batches", batchService.getAllBatches());
        categories.put("colleges", collegeService.getAllColleges());
        categories.put("faculties", facultyService.getAllFaculties());
        categories.put("majors", majorService.getAllMajors());
        categories.put("genders", genderService.getAllGenders());
        return ResponseEntity.ok(categories);
    }

    @GetMapping("/hierarchy")
    public ResponseEntity<Map<String, Object>> getCategoriesHierarchy() {
        Map<String, Object> hierarchy = new HashMap<>();
        List<CollegeDTO> colleges = collegeService.getAllColleges();

        Map<String, Object> collegeHierarchy = new HashMap<>();
        for (CollegeDTO college : colleges) {
            List<FacultyDTO> faculties = facultyService.getFacultiesByCollege(college.getCode());
            Map<String, Object> facultyHierarchy = new HashMap<>();

            for (FacultyDTO faculty : faculties) {
                List<MajorDTO> majors = majorService.getMajorsByFaculty(faculty.getCode());
                facultyHierarchy.put(faculty.getCode(), Map.of(
                    "faculty", faculty,
                    "majors", majors
                ));
            }

            collegeHierarchy.put(college.getCode(), Map.of(
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
