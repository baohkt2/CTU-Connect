package com.ctuconnect.model;

import java.util.UUID;

import lombok.Data;

@Data
public class UserModel {
    private UUID id;
    private String username;
    private String email;
    private String password;
    private boolean isActive;
}
