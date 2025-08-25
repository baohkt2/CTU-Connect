package com.ctuconnect.event;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserCreatedEvent {
    @JsonProperty("userId")
    private String userId;

    @JsonProperty("email")
    private String email;

    @JsonProperty("username")
    private String username;

    @JsonProperty("fullName")
    private String fullName;

    @JsonProperty("role")
    private String role;

    @JsonProperty("createdAt")
    private LocalDateTime createdAt;

    @JsonProperty("eventType")
    @Builder.Default
    private String eventType = "USER_CREATED";

    @JsonProperty("timestamp")
    @Builder.Default
    private LocalDateTime timestamp = LocalDateTime.now();
}
