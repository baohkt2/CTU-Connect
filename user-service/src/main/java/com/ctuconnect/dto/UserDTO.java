package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.Set;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UserDTO {
    private String id;
    private String email;
    private String username;
    private String fullName;
    private String role; // STUDENT / FACULTY / ADMIN
    private Boolean isActive;
    private Boolean isProfileCompleted;
    private String bio;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Set<String> friendIds;

    // Sinh viên
    private String studentId;

    // Giảng viên
    private String staffCode;
    private String position;       // Giảng viên, Trợ lý, ...
    private String academicTitle;  // Giáo sư, Phó GS, ...
    private String degree;         // Tiến sĩ, Thạc sĩ,...

    // Dành cho gửi dữ liệu (cập nhật thông tin người dùng) - sử dụng codes
    private String majorCode;
    private String facultyCode;
    private String collegeCode;
    private Integer batchYear;
    private String genderCode;

    // Dành cho hiển thị (khi truy vấn) - sử dụng names
    private String majorName;
    private String facultyName;
    private String collegeName;
    private String genderName;

    // Legacy fields cho backward compatibility
    private String major;
    private String faculty;
    private String college;
    private String gender;
    private String batch;

    private String avatarUrl;
    private String backgroundUrl;

    // Gợi ý bạn bè / so sánh
    private Integer mutualFriendsCount;
    private Boolean sameCollege;
    private Boolean sameFaculty;
    private Boolean sameMajor;
    private Boolean sameBatch;

    // Helper methods cho backward compatibility
    public String getMajor() {
        return majorName != null ? majorName : major;
    }

    public String getFaculty() {
        return facultyName != null ? facultyName : faculty;
    }

    public String getCollege() {
        return collegeName != null ? collegeName : college;
    }

    public String getGender() {
        return genderName != null ? genderName : gender;
    }

    public String getBatch() {
        return batchYear != null ? String.valueOf(batchYear) : batch;
    }

    // Setter methods cho backward compatibility
    public void setMajor(String major) {
        this.major = major;
        if (this.majorName == null) {
            this.majorName = major;
        }
    }

    public void setFaculty(String faculty) {
        this.faculty = faculty;
        if (this.facultyName == null) {
            this.facultyName = faculty;
        }
    }

    public void setCollege(String college) {
        this.college = college;
        if (this.collegeName == null) {
            this.collegeName = college;
        }
    }

    public void setGender(String gender) {
        this.gender = gender;
        if (this.genderName == null) {
            this.genderName = gender;
        }
    }

    public void setBatch(String batch) {
        this.batch = batch;
        if (this.batchYear == null && batch != null) {
            try {
                this.batchYear = Integer.valueOf(batch);
            } catch (NumberFormatException e) {
                // Ignore if batch is not a number
            }
        }
    }
}
