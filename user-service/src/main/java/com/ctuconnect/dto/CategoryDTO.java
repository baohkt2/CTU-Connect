package com.ctuconnect.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.util.List;

public class CategoryDTO {

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class CollegeInfo {
        private String name;
        private String code;
        private List<FacultyInfo> faculties;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class FacultyInfo {
        private String name;
        private String code;
        private CollegeBasicInfo college;
        private List<MajorInfo> majors;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class MajorInfo {
        private String name;
        private String code;
        private FacultyBasicInfo faculty;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class BatchInfo {
        private String year;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class GenderInfo {
        private String code;
        private String name;
    }

    // Basic info classes to avoid circular references
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class CollegeBasicInfo {
        private String name;
        private String code;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class FacultyBasicInfo {
        private String name;
        private String code;
        private CollegeBasicInfo college;
    }

    // Hierarchical structure for efficient frontend usage
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class HierarchicalCategories {
        private List<CollegeInfo> colleges;
        private List<BatchInfo> batches;
        private List<GenderInfo> genders;
        private List<PositionInfo> positions;
        private List<AcademicInfo> academics;
        private List<DegreeInfo> degrees;
    }

    @Builder
    @Data
    public static class PositionInfo {
        private String name;
        private String code;
    }

    @Builder
    @Data
    public static class AcademicInfo {
        private String name;
        private String code;
    }

    @Builder
    @Data
    public static class DegreeInfo {
        private String name;
        private String code;
    }
}
