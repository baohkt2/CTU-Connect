package com.ctuconnect.dto;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import com.fasterxml.jackson.annotation.JsonInclude;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class FriendRequestDTO {
    private String id;
    private String email;
    private String username;
    private String fullName;
    private String studentId;
    private String college;
    private String faculty;
    private String major;
    private Integer batch;
    private String gender;
    private Long mutualFriendsCount;
    private String requestType; // "SENT" or "RECEIVED"
}
