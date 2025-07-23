package com.ctuconnect.controller;

import com.ctuconnect.dto.CollegeDTO;
import com.ctuconnect.service.CollegeService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import jakarta.validation.Valid;

import java.util.List;

@RestController
@RequestMapping("/api/users/colleges")
@RequiredArgsConstructor
public class CollegeController {
    private final CollegeService collegeService;

    @GetMapping
    public ResponseEntity<List<CollegeDTO>> getAllColleges() {
        List<CollegeDTO> colleges = collegeService.getAllColleges();
        return ResponseEntity.ok(colleges);
    }

    @GetMapping("/{code}")
    public ResponseEntity<CollegeDTO> getCollegeByCode(@PathVariable String code) {
        return collegeService.getCollegeByCode(code)
                .map(college -> ResponseEntity.ok(college))
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<CollegeDTO> createCollege(@Valid @RequestBody CollegeDTO collegeDTO) {
        CollegeDTO createdCollege = collegeService.createCollege(collegeDTO);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdCollege);
    }

    @PutMapping("/{code}")
    public ResponseEntity<CollegeDTO> updateCollege(@PathVariable String code, @Valid @RequestBody CollegeDTO collegeDTO) {
        return collegeService.updateCollege(code, collegeDTO)
                .map(college -> ResponseEntity.ok(college))
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{code}")
    public ResponseEntity<Void> deleteCollege(@PathVariable String code) {
        if (collegeService.deleteCollege(code)) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}
