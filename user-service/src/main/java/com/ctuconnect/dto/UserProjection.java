package com.ctuconnect.dto;

/**
 * Projection interface for Cypher query results
 * Spring Data Neo4j will automatically map query results to this interface
 */
public interface UserProjection {
    String getId();

    String getEmail();

    String getUsername();

    String getStudentId();

    String getFullName();

    String getRole();

    String getBio();

    Boolean getIsActive();

    String getCreatedAt();

    String getUpdatedAt();
    String getCollege();

    String getFaculty();

    String getMajor();

    String getBatch();

    String getGender();

    Integer getFriendsCount();

    Integer getSentRequestsCount();

    Integer getReceivedRequestsCount();

    Integer getMutualFriendsCount();

    Boolean getSameCollege();

    Boolean getSameFaculty();

    Boolean getSameMajor();

}
