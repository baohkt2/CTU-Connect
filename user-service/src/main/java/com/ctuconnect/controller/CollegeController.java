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

    @GetMapping("/{name}")
    public ResponseEntity<CollegeDTO> getCollegeByName(@PathVariable String name) {
        return collegeService.getCollegeByName(name)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<CollegeDTO> createCollege(@Valid @RequestBody CollegeDTO collegeDTO) {
        CollegeDTO createdCollege = collegeService.createCollege(collegeDTO);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdCollege);
    }

    @PutMapping("/{name}")
    public ResponseEntity<CollegeDTO> updateCollege(@PathVariable String name, @Valid @RequestBody CollegeDTO collegeDTO) {
        return collegeService.updateCollege(name, collegeDTO)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{name}")
    public ResponseEntity<Void> deleteCollege(@PathVariable String name) {
        if (collegeService.deleteCollege(name)) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}
