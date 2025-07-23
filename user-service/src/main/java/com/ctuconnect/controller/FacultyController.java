package com.ctuconnect.controller;

import com.ctuconnect.dto.FacultyDTO;
import com.ctuconnect.service.FacultyService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import jakarta.validation.Valid;

import java.util.List;

@RestController
@RequestMapping("/api/users/faculties")
@RequiredArgsConstructor
public class FacultyController {
    private final FacultyService facultyService;

    @GetMapping
    public ResponseEntity<List<FacultyDTO>> getAllFaculties() {
        List<FacultyDTO> faculties = facultyService.getAllFaculties();
        return ResponseEntity.ok(faculties);
    }

    @GetMapping("/college/{collegeCode}")
    public ResponseEntity<List<FacultyDTO>> getFacultiesByCollege(@PathVariable String collegeCode) {
        List<FacultyDTO> faculties = facultyService.getFacultiesByCollege(collegeCode);
        return ResponseEntity.ok(faculties);
    }

    @GetMapping("/{code}")
    public ResponseEntity<FacultyDTO> getFacultyByCode(@PathVariable String code) {
        return facultyService.getFacultyByCode(code)
                .map(faculty -> ResponseEntity.ok(faculty))
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<FacultyDTO> createFaculty(@Valid @RequestBody FacultyDTO facultyDTO) {
        return facultyService.createFaculty(facultyDTO)
                .map(faculty -> ResponseEntity.status(HttpStatus.CREATED).body(faculty))
                .orElse(ResponseEntity.badRequest().build());
    }

    @PutMapping("/{code}")
    public ResponseEntity<FacultyDTO> updateFaculty(@PathVariable String code, @Valid @RequestBody FacultyDTO facultyDTO) {
        return facultyService.updateFaculty(code, facultyDTO)
                .map(faculty -> ResponseEntity.ok(faculty))
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{code}")
    public ResponseEntity<Void> deleteFaculty(@PathVariable String code) {
        if (facultyService.deleteFaculty(code)) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}
