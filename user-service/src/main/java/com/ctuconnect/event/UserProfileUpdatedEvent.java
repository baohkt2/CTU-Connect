package com.ctuconnect.event;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserProfileUpdatedEvent {
    private String userId;
    private String email;
    private String username;
    private String firstName;
    private String lastName;
    private String bio;
    private String profilePicture;
}
