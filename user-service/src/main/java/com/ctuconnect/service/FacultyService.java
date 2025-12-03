package com.ctuconnect.service;

import com.ctuconnect.dto.FacultyDTO;
import com.ctuconnect.entity.FacultyEntity;
import com.ctuconnect.repository.FacultyRepository;
import com.ctuconnect.repository.CollegeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import com.ctuconnect.entity.MajorEntity;

@Service
@RequiredArgsConstructor
public class FacultyService {
    private final FacultyRepository facultyRepository;
    private final CollegeRepository collegeRepository;

    public List<FacultyDTO> getAllFaculties() {
        return facultyRepository.findAllWithCollegeAndMajors().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public List<FacultyDTO> getFacultiesByCollege(String collegeName) {
        return facultyRepository.findByCollege(collegeName).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public Optional<FacultyDTO> getFacultyByName(String name) {
        return facultyRepository.findByNameWithCollegeAndMajors(name)
                .map(this::convertToDTO);
    }

    public Optional<FacultyDTO> createFaculty(FacultyDTO facultyDTO) {
        return collegeRepository.findById(facultyDTO.getCollegeName())
                .map(college -> {
                    FacultyEntity faculty = FacultyEntity.builder()
                            .name(facultyDTO.getName())
                            .code(facultyDTO.getCode())
                            .college(college)
                            .build();
                    FacultyEntity savedFaculty = facultyRepository.save(faculty);
                    return convertToDTO(savedFaculty);
                });
    }

    public Optional<FacultyDTO> updateFaculty(String name, FacultyDTO facultyDTO) {
        return facultyRepository.findById(name)
                .flatMap(existingFaculty ->
                    collegeRepository.findById(facultyDTO.getCollegeName())
                            .map(college -> {
                                existingFaculty.setCode(facultyDTO.getCode());
                                existingFaculty.setCollege(college);
                                FacultyEntity savedFaculty = facultyRepository.save(existingFaculty);
                                return convertToDTO(savedFaculty);
                            })
                );
    }

    public boolean deleteFaculty(String name) {
        if (facultyRepository.existsById(name)) {
            facultyRepository.deleteById(name);
            return true;
        }
        return false;
    }

    private FacultyDTO convertToDTO(FacultyEntity faculty) {
        String collegeName = null;
        if (faculty.getCollege() != null) {
            collegeName = faculty.getCollege().getName();
        }
        
        FacultyDTO dto = FacultyDTO.builder()
                .name(faculty.getName())
                .code(faculty.getCode())
                .collegeName(collegeName)
                .build();

        // Add majors if available
        if (faculty.getMajors() != null && !faculty.getMajors().isEmpty()) {
            List<String> majorNames = faculty.getMajors().stream()
                .map(MajorEntity::getName)
                .collect(Collectors.toList());
            dto.setMajors(majorNames);
        }

        return dto;
    }
}
