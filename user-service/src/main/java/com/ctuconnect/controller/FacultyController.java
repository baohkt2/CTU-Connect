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

    @GetMapping("/college/{collegeName}")
    public ResponseEntity<List<FacultyDTO>> getFacultiesByCollege(@PathVariable String collegeName) {
        List<FacultyDTO> faculties = facultyService.getFacultiesByCollege(collegeName);
        return ResponseEntity.ok(faculties);
    }

    @GetMapping("/{name}")
    public ResponseEntity<FacultyDTO> getFacultyByName(@PathVariable String name) {
        return facultyService.getFacultyByName(name)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<FacultyDTO> createFaculty(@Valid @RequestBody FacultyDTO facultyDTO) {
        return facultyService.createFaculty(facultyDTO)
                .map(faculty -> ResponseEntity.status(HttpStatus.CREATED).body(faculty))
                .orElse(ResponseEntity.badRequest().build());
    }

    @PutMapping("/{name}")
    public ResponseEntity<FacultyDTO> updateFaculty(@PathVariable String name, @Valid @RequestBody FacultyDTO facultyDTO) {
        return facultyService.updateFaculty(name, facultyDTO)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{name}")
    public ResponseEntity<Void> deleteFaculty(@PathVariable String name) {
        if (facultyService.deleteFaculty(name)) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}
