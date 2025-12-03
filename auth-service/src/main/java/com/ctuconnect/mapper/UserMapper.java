package com.ctuconnect.mapper;

import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.entity.UserEntity;

public class UserMapper {

    public static UserDTO toDto(UserEntity user) {
        return UserDTO.builder()
                .id(user.getId().toString())
                .email(user.getEmail())
                .fullName(user.getUsername()) // hoặc user.getFullName() nếu có
                .username(user.getUsername())
                .bio(null) // nếu có field thì map vào
                .isVerified(true) // nếu có logic kiểm tra thì thay vào
                .createdAt(user.getCreatedAt())
                .updatedAt(user.getUpdatedAt())
                .build();
    }
}
