package com.ctuconnect.controller;

import com.ctuconnect.dto.MajorDTO;
import com.ctuconnect.service.MajorService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import jakarta.validation.Valid;

import java.util.List;

@RestController
@RequestMapping("/api/users/majors")
@RequiredArgsConstructor
public class MajorController {
    private final MajorService majorService;

    @GetMapping
    public ResponseEntity<List<MajorDTO>> getAllMajors() {
        List<MajorDTO> majors = majorService.getAllMajors();
        return ResponseEntity.ok(majors);
    }

    @GetMapping("/faculty/{facultyName}")
    public ResponseEntity<List<MajorDTO>> getMajorsByFaculty(@PathVariable String facultyName) {
        List<MajorDTO> majors = majorService.getMajorsByFaculty(facultyName);
        return ResponseEntity.ok(majors);
    }

    @GetMapping("/{name}")
    public ResponseEntity<MajorDTO> getMajorByName(@PathVariable String name) {
        return majorService.getMajorByName(name)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<MajorDTO> createMajor(@Valid @RequestBody MajorDTO majorDTO) {
        return majorService.createMajor(majorDTO)
                .map(major -> ResponseEntity.status(HttpStatus.CREATED).body(major))
                .orElse(ResponseEntity.badRequest().build());
    }

    @PutMapping("/{name}")
    public ResponseEntity<MajorDTO> updateMajor(@PathVariable String name, @Valid @RequestBody MajorDTO majorDTO) {
        return majorService.updateMajor(name, majorDTO)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{name}")
    public ResponseEntity<Void> deleteMajor(@PathVariable String name) {
        if (majorService.deleteMajor(name)) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}
