package com.ctuconnect.service;

import com.ctuconnect.dto.FacultyDTO;
import com.ctuconnect.entity.CollegeEntity;
import com.ctuconnect.entity.FacultyEntity;
import com.ctuconnect.repository.CollegeRepository;
import com.ctuconnect.repository.FacultyRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class FacultyService {
    private final FacultyRepository facultyRepository;
    private final CollegeRepository collegeRepository;

    public List<FacultyDTO> getAllFaculties() {
        return facultyRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public List<FacultyDTO> getFacultiesByCollege(String collegeCode) {
        return facultyRepository.findByCollegeCode(collegeCode).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public Optional<FacultyDTO> getFacultyByCode(String code) {
        return facultyRepository.findById(code)
                .map(this::convertToDTO);
    }

    public Optional<FacultyDTO> createFaculty(FacultyDTO facultyDTO) {
        return collegeRepository.findById(facultyDTO.getCollegeCode())
                .map(college -> {
                    FacultyEntity faculty = FacultyEntity.builder()
                            .code(facultyDTO.getCode())
                            .name(facultyDTO.getName())
                            .college(college)
                            .build();
                    FacultyEntity savedFaculty = facultyRepository.save(faculty);
                    return convertToDTO(savedFaculty);
                });
    }

    public Optional<FacultyDTO> updateFaculty(String code, FacultyDTO facultyDTO) {
        return facultyRepository.findById(code)
                .flatMap(existingFaculty ->
                    collegeRepository.findById(facultyDTO.getCollegeCode())
                            .map(college -> {
                                existingFaculty.setName(facultyDTO.getName());
                                existingFaculty.setCollege(college);
                                FacultyEntity savedFaculty = facultyRepository.save(existingFaculty);
                                return convertToDTO(savedFaculty);
                            })
                );
    }

    public boolean deleteFaculty(String code) {
        if (facultyRepository.existsById(code)) {
            facultyRepository.deleteById(code);
            return true;
        }
        return false;
    }

    private FacultyDTO convertToDTO(FacultyEntity faculty) {
        return FacultyDTO.builder()
                .code(faculty.getCode())
                .name(faculty.getName())
                .collegeCode(faculty.getCollege() != null ? faculty.getCollege().getCode() : null)
                .collegeName(faculty.getCollege() != null ? faculty.getCollege().getName() : null)
                .build();
    }
}
