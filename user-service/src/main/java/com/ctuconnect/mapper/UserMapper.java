package com.ctuconnect.mapper;

import com.ctuconnect.dto.UserProfileDTO;
import com.ctuconnect.dto.UserSearchDTO;
import com.ctuconnect.dto.FriendRequestDTO;
import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.repository.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.stream.Collectors;

@Component
@Slf4j
public class UserMapper {

    public UserProfileDTO toUserProfileDTO(UserRepository.UserProfileProjection projection) {
        return UserProfileDTO.builder()
            .id(projection.getUserId())
            .email(projection.getEmail())
            .username(projection.getUsername())
            .studentId(projection.getStudentId())
            .fullName(projection.getFullName())
            .bio(projection.getBio())
            .role(projection.getRole())
            .isActive(projection.getIsActive())
            .createdAt(projection.getCreatedAt())
            .updatedAt(projection.getUpdatedAt())
            .college(projection.getCollege())
            .faculty(projection.getFaculty())
            .major(projection.getMajor())
            .batch(projection.getBatch())
            .gender(projection.getGender())
            .friendsCount(projection.getFriendsCount())
            .sentRequestsCount(projection.getSentRequestsCount())
            .receivedRequestsCount(projection.getReceivedRequestsCount())
            .build();
    }

    public UserSearchDTO toUserSearchDTO(UserRepository.UserSearchProjection projection) {
        UserEntity user = projection.getUser();

        return UserSearchDTO.builder()
            .id(user.getId())
            .email(user.getEmail())
            .username(user.getUsername())
            .studentId(user.getStudentId())
            .fullName(user.getFullName())
            .role(user.getRole())
            .isActive(user.getIsActive())
            .college(projection.getCollege())
            .faculty(projection.getFaculty())
            .major(projection.getMajor())
            .batch(projection.getBatch())
            .gender(projection.getGender())
            .friendsCount(projection.getFriendsCount())
            .mutualFriendsCount(projection.getMutualFriendsCount())
            .isFriend(projection.getIsFriend())
            .requestSent(projection.getRequestSent())
            .requestReceived(projection.getRequestReceived())
            .sameCollege(user.getMajor() != null && projection.getCollege() != null)
            .sameFaculty(user.getMajor() != null && projection.getFaculty() != null)
            .sameMajor(user.getMajor() != null && projection.getMajor() != null)
            .sameBatch(user.getBatch() != null && projection.getBatch() != null)
            .build();
    }

    public FriendRequestDTO toFriendRequestDTO(UserRepository.FriendRequestProjection projection) {
        UserEntity user = projection.getUser();

        return FriendRequestDTO.builder()
            .id(user.getId())
            .email(user.getEmail())
            .username(user.getUsername())
            .fullName(user.getFullName())
            .studentId(user.getStudentId())
            .college(projection.getCollege())
            .faculty(projection.getFaculty())
            .major(projection.getMajor())
            .batch(projection.getBatch())
            .gender(projection.getGender())
            .mutualFriendsCount(projection.getMutualFriendsCount())
            .requestType(projection.getRequestType())
            .build();
    }

    public UserProfileDTO toUserProfileDTO(UserEntity user) {
        log.debug("Mapping UserEntity to UserProfileDTO - avatarUrl: {}, backgroundUrl: {}", 
                user.getAvatarUrl(), user.getBackgroundUrl());
        
        return UserProfileDTO.builder()
            .id(user.getId())
            .email(user.getEmail())
            .username(user.getUsername())
            .studentId(user.getStudentId())
            .fullName(user.getFullName())
            .bio(user.getBio())
            .role(user.getRole())
            .isActive(user.getIsActive())
            .createdAt(user.getCreatedAt())
            .updatedAt(user.getUpdatedAt())
            .avatarUrl(user.getAvatarUrl())
            .backgroundUrl(user.getBackgroundUrl())
            .college(user.getMajor() != null && user.getMajor().getFaculty() != null &&
                    user.getMajor().getFaculty().getCollege() != null ?
                    user.getMajor().getFaculty().getCollege().getName() : null)
            .faculty(user.getMajor() != null && user.getMajor().getFaculty() != null ?
                    user.getMajor().getFaculty().getName() : null)
            .major(user.getMajor() != null ? user.getMajor().getName() : null)
            .batch(user.getBatch() != null ? user.getBatch().getYear() : null)
            .gender(user.getGender() != null ? user.getGender().getName() : null)
            .collegeCode(user.getMajor() != null && user.getMajor().getFaculty() != null &&
                    user.getMajor().getFaculty().getCollege() != null ?
                    user.getMajor().getFaculty().getCollege().getCode() : null)
            .facultyCode(user.getMajor() != null && user.getMajor().getFaculty() != null ?
                    user.getMajor().getFaculty().getCode() : null)
            .majorCode(user.getMajor() != null ? user.getMajor().getCode() : null)
            .batchCode(user.getBatch() != null ? user.getBatch().getYear() : null)
            .genderCode(user.getGender() != null ? user.getGender().getCode() : null)
            .friendsCount((long) user.getFriends().size())
            .sentRequestsCount((long) user.getSentFriendRequests().size())
            .receivedRequestsCount((long) user.getReceivedFriendRequests().size())
            .build();
    }

    public UserSearchDTO toUserSearchDTO(UserEntity user) {
        return UserSearchDTO.builder()
            .id(user.getId())
            .email(user.getEmail())
            .username(user.getUsername())
            .studentId(user.getStudentId())
            .fullName(user.getFullName())
            .role(user.getRole())
            .isActive(user.getIsActive())
            .college(user.getMajor() != null && user.getMajor().getFaculty() != null &&
                    user.getMajor().getFaculty().getCollege() != null ?
                    user.getMajor().getFaculty().getCollege().getName() : null)
            .faculty(user.getMajor() != null && user.getMajor().getFaculty() != null ?
                    user.getMajor().getFaculty().getName() : null)
            .major(user.getMajor() != null ? user.getMajor().getName() : null)
            .batch(user.getBatch() != null ? user.getBatch().getYear() : null)
            .gender(user.getGender() != null ? user.getGender().getName() : null)
            .friendsCount((long) user.getFriends().size())
            .mutualFriendsCount(0L)
            .isFriend(false)
            .requestSent(false)
            .requestReceived(false)
            .sameCollege(false)
            .sameFaculty(false)
            .sameMajor(false)
            .sameBatch(false)
            .build();
    }

    public UserDTO toUserDTO(UserRepository.UserSearchProjection projection) {
        UserEntity user = projection.getUser();
        
        UserDTO dto = new UserDTO();
        dto.setId(user.getId());
        dto.setEmail(user.getEmail());
        dto.setUsername(user.getUsername());
        dto.setStudentId(user.getStudentId());
        dto.setFullName(user.getFullName());
        dto.setRole(user.getRole());
        dto.setBio(user.getBio());
        dto.setIsActive(user.getIsActive());
        dto.setCreatedAt(user.getCreatedAt());
        dto.setUpdatedAt(user.getUpdatedAt());
        dto.setCollege(projection.getCollege());
        dto.setFaculty(projection.getFaculty());
        dto.setMajor(projection.getMajor());
        dto.setBatch(projection.getBatch());
        dto.setGender(projection.getGender());
        dto.setFriendIds(user.getFriends().stream()
            .map(UserEntity::getId)
            .collect(Collectors.toSet()));
        dto.setMutualFriendsCount(projection.getMutualFriendsCount() != null ? 
            projection.getMutualFriendsCount().intValue() : 0);
        dto.setSameCollege(projection.getCollege() != null);
        dto.setSameFaculty(projection.getFaculty() != null);
        dto.setSameMajor(projection.getMajor() != null);
        
        return dto;
    }

    public UserDTO toUserDTO(UserEntity user) {
        UserDTO dto = new UserDTO();
        dto.setId(user.getId());
        dto.setEmail(user.getEmail());
        dto.setUsername(user.getUsername());
        dto.setStudentId(user.getStudentId());
        dto.setFullName(user.getFullName());
        dto.setRole(user.getRole());
        dto.setBio(user.getBio());
        dto.setIsActive(user.getIsActive());
        dto.setCreatedAt(user.getCreatedAt());
        dto.setUpdatedAt(user.getUpdatedAt());
        dto.setCollege(user.getMajor() != null && user.getMajor().getFaculty() != null &&
                user.getMajor().getFaculty().getCollege() != null ?
                user.getMajor().getFaculty().getCollege().getName() : null);
        dto.setFaculty(user.getMajor() != null && user.getMajor().getFaculty() != null ?
                user.getMajor().getFaculty().getName() : null);
        dto.setMajor(user.getMajor() != null ? user.getMajor().getName() : null);
        dto.setBatch(user.getBatch() != null ? user.getBatch().getYear() : null);
        dto.setGender(user.getGender() != null ? user.getGender().getName() : null);
        dto.setFriendIds(user.getFriends().stream()
            .map(UserEntity::getId)
            .collect(Collectors.toSet()));
        dto.setMutualFriendsCount(0);
        dto.setSameCollege(false);
        dto.setSameFaculty(false);
        dto.setSameMajor(false);
        
        return dto;
    }
}
