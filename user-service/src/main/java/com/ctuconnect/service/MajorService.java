package com.ctuconnect.service;

import com.ctuconnect.dto.MajorDTO;
import com.ctuconnect.entity.FacultyEntity;
import com.ctuconnect.entity.MajorEntity;
import com.ctuconnect.repository.FacultyRepository;
import com.ctuconnect.repository.MajorRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class MajorService {
    private final MajorRepository majorRepository;
    private final FacultyRepository facultyRepository;

    public List<MajorDTO> getAllMajors() {
        return majorRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public List<MajorDTO> getMajorsByFaculty(String facultyCode) {
        return majorRepository.findByFacultyCode(facultyCode).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public Optional<MajorDTO> getMajorByCode(String code) {
        return majorRepository.findById(code)
                .map(this::convertToDTO);
    }

    public Optional<MajorDTO> createMajor(MajorDTO majorDTO) {
        return facultyRepository.findById(majorDTO.getFacultyCode())
                .map(faculty -> {
                    MajorEntity major = MajorEntity.builder()
                            .code(majorDTO.getCode())
                            .name(majorDTO.getName())
                            .faculty(faculty)
                            .build();
                    MajorEntity savedMajor = majorRepository.save(major);
                    return convertToDTO(savedMajor);
                });
    }

    public Optional<MajorDTO> updateMajor(String code, MajorDTO majorDTO) {
        return majorRepository.findById(code)
                .flatMap(existingMajor ->
                    facultyRepository.findById(majorDTO.getFacultyCode())
                            .map(faculty -> {
                                existingMajor.setName(majorDTO.getName());
                                existingMajor.setFaculty(faculty);
                                MajorEntity savedMajor = majorRepository.save(existingMajor);
                                return convertToDTO(savedMajor);
                            })
                );
    }

    public boolean deleteMajor(String code) {
        if (majorRepository.existsById(code)) {
            majorRepository.deleteById(code);
            return true;
        }
        return false;
    }

    private MajorDTO convertToDTO(MajorEntity major) {
        return MajorDTO.builder()
                .code(major.getCode())
                .name(major.getName())
                .facultyCode(major.getFaculty() != null ? major.getFaculty().getCode() : null)
                .facultyName(major.getFaculty() != null ? major.getFaculty().getName() : null)
                .collegeCode(major.getFaculty() != null && major.getFaculty().getCollege() != null ?
                           major.getFaculty().getCollege().getCode() : null)
                .collegeName(major.getFaculty() != null && major.getFaculty().getCollege() != null ?
                           major.getFaculty().getCollege().getName() : null)
                .build();
    }
}
