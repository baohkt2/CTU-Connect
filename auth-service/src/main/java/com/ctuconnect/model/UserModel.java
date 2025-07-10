package com.ctuconnect.model;

import lombok.Data;

@Data
public class UserModel {
    private String id;
    private String username;
    private String email;
    private String password;
    private boolean isActive;
}
