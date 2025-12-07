package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * User academic profile for Python model service
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserAcademicProfile implements Serializable {
    private String userId;
    private String major;
    private String faculty;
    private String degree;
    private String batch;
    private String studentId;
}
