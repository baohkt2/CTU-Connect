package com.ctuconnect.mapper;

import com.ctuconnect.dto.UserProfileDTO;
import com.ctuconnect.dto.UserSearchDTO;
import com.ctuconnect.dto.FriendRequestDTO;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.repository.UserRepository;
import org.springframework.stereotype.Component;

@Component
public class UserMapper {

    public UserProfileDTO toUserProfileDTO(UserRepository.UserProfileProjection projection) {
        UserEntity user = projection.getUser();

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
            .college(user.getMajor() != null && user.getMajor().getFaculty() != null &&
                    user.getMajor().getFaculty().getCollege() != null ?
                    user.getMajor().getFaculty().getCollege().getName() : null)
            .faculty(user.getMajor() != null && user.getMajor().getFaculty() != null ?
                    user.getMajor().getFaculty().getName() : null)
            .major(user.getMajor() != null ? user.getMajor().getName() : null)
            .batch(user.getBatch() != null ? user.getBatch().getYear() : null)
            .gender(user.getGender() != null ? user.getGender().getName() : null)
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
}
