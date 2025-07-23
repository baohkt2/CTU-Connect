package com.ctuconnect.service;

import com.ctuconnect.dto.CollegeDTO;
import com.ctuconnect.entity.CollegeEntity;
import com.ctuconnect.repository.CollegeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class CollegeService {
    private final CollegeRepository collegeRepository;

    public List<CollegeDTO> getAllColleges() {
        return collegeRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public Optional<CollegeDTO> getCollegeByCode(String code) {
        return collegeRepository.findById(code)
                .map(this::convertToDTO);
    }

    public CollegeDTO createCollege(CollegeDTO collegeDTO) {
        CollegeEntity college = CollegeEntity.builder()
                .code(collegeDTO.getCode())
                .name(collegeDTO.getName())
                .build();
        CollegeEntity savedCollege = collegeRepository.save(college);
        return convertToDTO(savedCollege);
    }

    public Optional<CollegeDTO> updateCollege(String code, CollegeDTO collegeDTO) {
        return collegeRepository.findById(code)
                .map(existingCollege -> {
                    existingCollege.setName(collegeDTO.getName());
                    CollegeEntity savedCollege = collegeRepository.save(existingCollege);
                    return convertToDTO(savedCollege);
                });
    }

    public boolean deleteCollege(String code) {
        if (collegeRepository.existsById(code)) {
            collegeRepository.deleteById(code);
            return true;
        }
        return false;
    }

    private CollegeDTO convertToDTO(CollegeEntity college) {
        return CollegeDTO.builder()
                .code(college.getCode())
                .name(college.getName())
                .build();
    }
}
