package com.ctuconnect.service;

import com.ctuconnect.dto.GenderDTO;
import com.ctuconnect.entity.GenderEntity;
import com.ctuconnect.repository.GenderRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class GenderService {
    private final GenderRepository genderRepository;

    public List<GenderDTO> getAllGenders() {
        return genderRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public Optional<GenderDTO> getGenderByCode(String code) {
        return genderRepository.findById(code)
                .map(this::convertToDTO);
    }

    public GenderDTO createGender(GenderDTO genderDTO) {
        GenderEntity gender = GenderEntity.builder()
                .code(genderDTO.getCode())
                .name(genderDTO.getName())
                .build();
        GenderEntity savedGender = genderRepository.save(gender);
        return convertToDTO(savedGender);
    }

    public Optional<GenderDTO> updateGender(String code, GenderDTO genderDTO) {
        return genderRepository.findById(code)
                .map(existingGender -> {
                    existingGender.setName(genderDTO.getName());
                    GenderEntity savedGender = genderRepository.save(existingGender);
                    return convertToDTO(savedGender);
                });
    }

    public boolean deleteGender(String code) {
        if (genderRepository.existsById(code)) {
            genderRepository.deleteById(code);
            return true;
        }
        return false;
    }

    private GenderDTO convertToDTO(GenderEntity gender) {
        return GenderDTO.builder()
                .code(gender.getCode())
                .name(gender.getName())
                .build();
    }
}
