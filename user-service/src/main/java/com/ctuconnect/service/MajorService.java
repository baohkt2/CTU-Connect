package com.ctuconnect.service;

import com.ctuconnect.dto.MajorDTO;
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

    public List<MajorDTO> getMajorsByFaculty(String facultyName) {
        return majorRepository.findByName(facultyName).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public Optional<MajorDTO> getMajorByName(String name) {
        return majorRepository.findById(name)
                .map(this::convertToDTO);
    }

    public Optional<MajorDTO> createMajor(MajorDTO majorDTO) {
        return facultyRepository.findById(majorDTO.getFacultyName())
                .map(faculty -> {
                    MajorEntity major = MajorEntity.builder()
                            .name(majorDTO.getName())
                            .code(majorDTO.getCode())
                            .faculty(majorDTO.getFacultyName())
                            .facultyEntity(faculty)
                            .build();
                    MajorEntity savedMajor = majorRepository.save(major);
                    return convertToDTO(savedMajor);
                });
    }

    public Optional<MajorDTO> updateMajor(String name, MajorDTO majorDTO) {
        return majorRepository.findById(name)
                .flatMap(existingMajor ->
                    facultyRepository.findById(majorDTO.getFacultyName())
                            .map(faculty -> {
                                existingMajor.setCode(majorDTO.getCode());
                                existingMajor.setFaculty(majorDTO.getFacultyName());
                                existingMajor.setFacultyEntity(faculty);
                                MajorEntity savedMajor = majorRepository.save(existingMajor);
                                return convertToDTO(savedMajor);
                            })
                );
    }

    public boolean deleteMajor(String name) {
        if (majorRepository.existsById(name)) {
            majorRepository.deleteById(name);
            return true;
        }
        return false;
    }

    private MajorDTO convertToDTO(MajorEntity major) {
        String collegeName = null;
        if (major.getFacultyEntity() != null && major.getFacultyEntity().getCollege() != null) {
            collegeName = major.getFacultyEntity().getCollege();
        }

        return MajorDTO.builder()
                .name(major.getName())
                .code(major.getCode())
                .facultyName(major.getFaculty())
                .collegeName(collegeName)
                .build();
    }
}
