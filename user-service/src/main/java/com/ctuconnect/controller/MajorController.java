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

    @GetMapping("/faculty/{facultyCode}")
    public ResponseEntity<List<MajorDTO>> getMajorsByFaculty(@PathVariable String facultyCode) {
        List<MajorDTO> majors = majorService.getMajorsByFaculty(facultyCode);
        return ResponseEntity.ok(majors);
    }

    @GetMapping("/{code}")
    public ResponseEntity<MajorDTO> getMajorByCode(@PathVariable String code) {
        return majorService.getMajorByCode(code)
                .map(major -> ResponseEntity.ok(major))
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<MajorDTO> createMajor(@Valid @RequestBody MajorDTO majorDTO) {
        return majorService.createMajor(majorDTO)
                .map(major -> ResponseEntity.status(HttpStatus.CREATED).body(major))
                .orElse(ResponseEntity.badRequest().build());
    }

    @PutMapping("/{code}")
    public ResponseEntity<MajorDTO> updateMajor(@PathVariable String code, @Valid @RequestBody MajorDTO majorDTO) {
        return majorService.updateMajor(code, majorDTO)
                .map(major -> ResponseEntity.ok(major))
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{code}")
    public ResponseEntity<Void> deleteMajor(@PathVariable String code) {
        if (majorService.deleteMajor(code)) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}
