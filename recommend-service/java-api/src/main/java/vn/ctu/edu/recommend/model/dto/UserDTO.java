package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO for User data from user-service
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserDTO {
    private String id;
    private String username;
    private String email;
    private String fullName;
    private String avatar;
    private String bio;
    private String major;
    private String faculty;
    private String degree;
    private String batch;
    private String studentId;
    private Boolean isActive;
}
