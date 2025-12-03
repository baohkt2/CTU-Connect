package com.ctuconnect.controller;

import com.ctuconnect.dto.GenderDTO;
import com.ctuconnect.service.GenderService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import jakarta.validation.Valid;

import java.util.List;

@RestController
@RequestMapping("/api/users/genders")
@RequiredArgsConstructor
public class GenderController {
    private final GenderService genderService;

    @GetMapping
    public ResponseEntity<List<GenderDTO>> getAllGenders() {
        List<GenderDTO> genders = genderService.getAllGenders();
        return ResponseEntity.ok(genders);
    }

    @GetMapping("/{code}")
    public ResponseEntity<GenderDTO> getGenderByCode(@PathVariable String code) {
        return genderService.getGenderByCode(code)
                .map(gender -> ResponseEntity.ok(gender))
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<GenderDTO> createGender(@Valid @RequestBody GenderDTO genderDTO) {
        GenderDTO createdGender = genderService.createGender(genderDTO);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdGender);
    }

    @PutMapping("/{code}")
    public ResponseEntity<GenderDTO> updateGender(@PathVariable String code, @Valid @RequestBody GenderDTO genderDTO) {
        return genderService.updateGender(code, genderDTO)
                .map(gender -> ResponseEntity.ok(gender))
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{code}")
    public ResponseEntity<Void> deleteGender(@PathVariable String code) {
        if (genderService.deleteGender(code)) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}
