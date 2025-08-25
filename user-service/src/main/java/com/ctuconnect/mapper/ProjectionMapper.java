package com.ctuconnect.mapper;

import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.dto.UserProjection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Mapper for converting UserProjection (from Cypher queries) to UserDTO
 * This eliminates the N+1 query problem by working with pre-loaded data only
 */
@Component
public class ProjectionMapper {

    private static final Logger logger = LoggerFactory.getLogger(ProjectionMapper.class);

    /**
     * Convert UserProjection to UserDTO
     * No database queries - works with data already loaded by Cypher projection
     */
    public UserDTO projectionToUserDTO(UserProjection projection) {
        if (projection == null) {
            logger.warn("Attempting to map null UserProjection to UserDTO");
            return null;
        }

        try {
            UserDTO dto = new UserDTO();

            // Basic user information mapping
            dto.setId(projection.getId());
            dto.setEmail(projection.getEmail());
            dto.setUsername(safeGetString(projection.getUsername()));
            dto.setStudentId(projection.getStudentId());
            dto.setFullName(projection.getFullName());
            dto.setRole(projection.getRole());
            dto.setBio(projection.getBio());
            dto.setIsActive(projection.getIsActive());

            // University structure information (already loaded by Cypher query)
            dto.setCollege(projection.getCollege());
            dto.setFaculty(projection.getFaculty());
            dto.setMajor(projection.getMajor());
            dto.setBatch(projection.getBatch());
            dto.setGender(projection.getGender());

            // Handle datetime parsing safely
            dto.setCreatedAt(parseDateTime(projection.getCreatedAt()));
            dto.setUpdatedAt(parseDateTime(projection.getUpdatedAt()));

            // Relationship analysis (calculated by Cypher query)
            dto.setMutualFriendsCount(projection.getMutualFriendsCount() != null ? projection.getMutualFriendsCount() : 0);
            dto.setSameCollege(projection.getSameCollege() != null ? projection.getSameCollege() : false);
            dto.setSameFaculty(projection.getSameFaculty() != null ? projection.getSameFaculty() : false);
            dto.setSameMajor(projection.getSameMajor() != null ? projection.getSameMajor() : false);

            return dto;

        } catch (Exception e) {
            logger.error("Error mapping UserProjection to UserDTO for user {}: {}",
                        projection.getId(), e.getMessage(), e);
            return null; // Return null instead of throwing exception to prevent cascade failures
        }
    }

    /**
     * Convert list of UserProjection to list of UserDTO
     * Filters out any null results from failed mappings
     */
    public List<UserDTO> projectionsToUserDTOs(List<UserProjection> projections) {
        if (projections == null) {
            logger.warn("Attempting to map null list of UserProjection to UserDTO");
            return null;
        }

        try {
            return projections.stream()
                    .map(this::projectionToUserDTO)
                    .filter(dto -> dto != null) // Filter out failed mappings
                    .collect(Collectors.toList());
        } catch (Exception e) {
            logger.error("Error mapping list of UserProjection to UserDTO: {}", e.getMessage(), e);
            return null;
        }
    }

    /**
     * Safe getter for String values - handles null and empty strings
     */
    private String safeGetString(String value) {
        if (value == null || value.trim().isEmpty()) {
            return null;
        }
        return value.trim();
    }

    /**
     * Parse datetime string safely with multiple format support
     */
    private LocalDateTime parseDateTime(String dateTimeStr) {
        if (dateTimeStr == null || dateTimeStr.trim().isEmpty()) {
            return null;
        }

        try {
            // Remove 'Z' suffix if present (ISO format with timezone)
            String cleanedStr = dateTimeStr.replace("Z", "");

            // Try parsing standard LocalDateTime format
            if (cleanedStr.contains("T")) {
                return LocalDateTime.parse(cleanedStr);
            }

            // Try parsing without 'T' separator
            return LocalDateTime.parse(cleanedStr);

        } catch (Exception e) {
            logger.warn("Failed to parse datetime string '{}': {}", dateTimeStr, e.getMessage());
            return null; // Return null instead of throwing exception
        }
    }
}
