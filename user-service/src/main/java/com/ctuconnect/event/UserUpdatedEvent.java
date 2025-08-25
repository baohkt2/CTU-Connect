package com.ctuconnect.event;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.LocalDateTime;
import java.time.OffsetDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserUpdatedEvent {
    @JsonProperty("userId")
    private String userId;

    @JsonProperty("email")
    private String email;

    @JsonProperty("username")
    private String username;

    @JsonProperty("fullName")
    private String fullName;

    @JsonProperty("bio")
    private String bio;

    @JsonProperty("studentId")
    private String studentId;

    @JsonProperty("role")
    private String role;

    @JsonProperty("isActive")
    private Boolean isActive;

    @JsonProperty("updatedAt")
    private OffsetDateTime updatedAt;

    @JsonProperty("eventType")
    @Builder.Default
    private String eventType = "USER_UPDATED";

    @JsonProperty("timestamp")
    @Builder.Default
    private OffsetDateTime timestamp = OffsetDateTime.now();
}
