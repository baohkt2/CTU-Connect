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

    public Optional<CollegeDTO> getCollegeByName(String name) {
        return collegeRepository.findById(name)
                .map(this::convertToDTO);
    }

    public CollegeDTO createCollege(CollegeDTO collegeDTO) {
        CollegeEntity college = CollegeEntity.builder()
                .name(collegeDTO.getName())
                .code(collegeDTO.getCode())
                .build();
        CollegeEntity savedCollege = collegeRepository.save(college);
        return convertToDTO(savedCollege);
    }

    public Optional<CollegeDTO> updateCollege(String name, CollegeDTO collegeDTO) {
        return collegeRepository.findById(name)
                .map(existingCollege -> {
                    existingCollege.setCode(collegeDTO.getCode());
                    // name không được thay đổi vì là ID
                    CollegeEntity savedCollege = collegeRepository.save(existingCollege);
                    return convertToDTO(savedCollege);
                });
    }

    public boolean deleteCollege(String name) {
        if (collegeRepository.existsById(name)) {
            collegeRepository.deleteById(name);
            return true;
        }
        return false;
    }

    private CollegeDTO convertToDTO(CollegeEntity college) {
        return CollegeDTO.builder()
                .name(college.getName())
                .code(college.getCode())
                .build();
    }
}
