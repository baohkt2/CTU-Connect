package com.ctuconnect.controller;

import com.ctuconnect.dto.CategoryDTO;
import com.ctuconnect.service.CategoryService;
import com.ctuconnect.security.annotation.RequireAuth;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users/categories")
@RequiredArgsConstructor
public class CategoriesController {

    private final CategoryService categoryService;

    @GetMapping("/all")
    @RequireAuth // Get all categories in hierarchical structure
    public ResponseEntity<CategoryDTO.HierarchicalCategories> getAllCategories() {
        return ResponseEntity.ok(categoryService.getAllCategoriesHierarchical());
    }

   /* @GetMapping("/positions")
    public ResponseEntity<List<CategoryDTO.PositionInfo>> getAllPositions() {
        return ResponseEntity.ok(categoryService.getAllPositions());
    }

    @GetMapping("/academic-titles")
    public ResponseEntity<List<CategoryDTO.AcademicTitleInfo>> getAllAcademicTitles() {
        return ResponseEntity.ok(categoryService.getAllAcademicTitles());
    }

    @GetMapping("/degrees")
    public ResponseEntity<List<CategoryDTO.DegreeInfo>> getAllDegrees() {
        return ResponseEntity.ok(categoryService.getAllDegrees());
    }*/

    @GetMapping("/colleges")
    @RequireAuth // Get all colleges with their faculties and majors
    public ResponseEntity<java.util.List<CategoryDTO.CollegeInfo>> getColleges() {
        return ResponseEntity.ok(categoryService.getAllColleges());
    }

    @GetMapping("/faculties")
    @RequireAuth // Get all faculties with their majors
    public ResponseEntity<java.util.List<CategoryDTO.FacultyInfo>> getFaculties() {
        return ResponseEntity.ok(categoryService.getAllFaculties());
    }

    @GetMapping("/majors")
    @RequireAuth // Get all majors
    public ResponseEntity<java.util.List<CategoryDTO.MajorInfo>> getMajors() {
        return ResponseEntity.ok(categoryService.getAllMajors());
    }

    @GetMapping("/batches")
    @RequireAuth // Get all batches
    public ResponseEntity<java.util.List<CategoryDTO.BatchInfo>> getBatches() {
        return ResponseEntity.ok(categoryService.getAllBatches());
    }

    @GetMapping("/genders")
    @RequireAuth // Get all genders
    public ResponseEntity<java.util.List<CategoryDTO.GenderInfo>> getGenders() {
        return ResponseEntity.ok(categoryService.getAllGenders());
    }
}
