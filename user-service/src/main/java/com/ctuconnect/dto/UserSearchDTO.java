package com.ctuconnect.dto;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class UserSearchDTO {
    private String id;
    private String email;
    private String username;
    private String studentId;
    private String fullName;
    private String role;
    private Boolean isActive;

    // Academic information
    private String college;
    private String faculty;
    private String major;
    private Integer batch;
    private String gender;

    // Social information
    private Long friendsCount;
    private Long mutualFriendsCount;

    // Relationship flags
    private Boolean isFriend;
    private Boolean requestSent;
    private Boolean requestReceived;
    private Boolean sameCollege;
    private Boolean sameFaculty;
    private Boolean sameMajor;
    private Boolean sameBatch;
}
