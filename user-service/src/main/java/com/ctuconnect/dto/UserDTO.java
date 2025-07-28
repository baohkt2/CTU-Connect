package com.ctuconnect.dto;

import com.ctuconnect.entity.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
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
    private Info major;
    private BatchInfo batch;
    // Giảng viên
    private String staffCode;
    private Info position;
    private Info academic;
    private Info degree;
    // Thông tin chung
    private Info faculty;
    private Info college;
    private Info gender;
    private String avatarUrl;
    private String backgroundUrl;
    // Gợi ý bạn bè / so sánh
    private Integer mutualFriendsCount;
    private Boolean sameCollege;
    private Boolean sameFaculty;
    private Boolean sameMajor;
    private Boolean sameBatch;

    public String getMajorId() {
        return major != null ? major.getCode() : null;
    }

    public String getBatchId() {
        return batch != null ? batch.getYear() : null;
    }

    public String getCollegeId() {
        return college != null ? college.getCode() : null;
    }

    public String getFacultyId() {
        return faculty != null ? faculty.getCode() : null;
    }

    public String getGenderId() {
        return gender != null ? gender.getCode() : null;
    }

    public String getPositionId() {
        return position != null ? position.getCode() : null;
    }

    public String getAcademicId() {
        return academic != null ? academic.getCode() : null;
    }

    public String getDegreeId() {
        return degree != null ? degree.getCode() : null;
    }

    public void setMajor(MajorEntity major) {
        if (major != null) {
            this.major = Info.builder()
                    .name(major.getName())
                    .code(major.getCode())
                    .build();
        } else {
            this.major = null;
        }
    }

    public void setBatch(BatchEntity batch) {
        if (batch != null) {
            this.batch = BatchInfo.builder()
                    .year(batch.getYear())
                    .build();
        } else {
            this.batch = null;
        }
    }

    public void setCollege(CollegeEntity college) {
        if (college != null) {
            this.college = Info.builder()
                    .name(college.getName())
                    .code(college.getCode())
                    .build();
        } else {
            this.college = null;
        }
    }

    public void setFaculty(FacultyEntity faculty) {
        if (faculty != null) {
            this.faculty = Info.builder()
                    .name(faculty.getName())
                    .code(faculty.getCode())
                    .build();
        } else {
            this.faculty = null;
        }
    }

    public void setGender(GenderEntity gender) {
        if (gender != null) {
            this.gender = Info.builder()
                    .name(gender.getName())
                    .code(gender.getCode())
                    .build();
        } else {
            this.gender = null;
        }
    }

    public void setPosition(PositionEntity position) {
        if (position != null) {
            this.position = Info.builder()
                    .name(position.getName())
                    .code(position.getCode())
                    .build();
        } else {
            this.position = null;
        }
    }

    public void setAcademic(AcademicEntity academic) {
        if (academic != null) {
            this.academic = Info.builder()
                    .name(academic.getName())
                    .code(academic.getCode())
                    .build();
        } else {
            this.academic = null;
        }
    }

    public void setDegree(DegreeEntity degree) {
        if (degree != null) {
            this.degree = Info.builder()
                    .name(degree.getName())
                    .code(degree.getCode())
                    .build();
        } else {
            this.degree = null;
        }
    }

    @Builder
    @Data
    public static class Info {
        private String name;
        private String code;
    }

    @Data
    @Builder
    public static class BatchInfo {
        private String year;
    }


}
