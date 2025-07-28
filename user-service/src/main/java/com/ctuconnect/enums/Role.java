package com.ctuconnect.enums;

import lombok.Getter;

@Getter
public enum Role {
    STUDENT("STUDENT"),
    LECTURER("LECTURER"),
    ADMIN("ADMIN"),
    USER("USER");

    private final String code;

    Role(String code) {
        this.code = code;
    }

    public static Role fromString(String value) {
        for (Role r : Role.values()) {
            if (r.name().equalsIgnoreCase(value) || r.code.equalsIgnoreCase(value)) {
                return r;
            }
        }
        throw new IllegalArgumentException("Invalid role: " + value);
    }

    public static Role fromEmail(String email) {
        if (email == null || email.isBlank()) {
            return STUDENT; // fallback
        }

        String lowerEmail = email.toLowerCase();

        if (lowerEmail.contains("@ctu.edu.vn")) {
            // Có thể refine sâu hơn nếu có phân biệt giữa faculty & student qua prefix email
            if (lowerEmail.startsWith("gv.") || lowerEmail.contains(".gv@")) {
                return LECTURER;
            }
            if (lowerEmail.contains("admin") || lowerEmail.startsWith("admin.")) {
                return ADMIN;
            }
            return STUDENT;
        }

        return STUDENT; // default fallback
    }
}
