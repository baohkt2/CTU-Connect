package com.ctuconnect.exception;

/**
 * Custom exception for user mapping errors
 */
public class UserMappingException extends RuntimeException {

    public UserMappingException(String message) {
        super(message);
    }

    public UserMappingException(String message, Throwable cause) {
        super(message, cause);
    }

    public UserMappingException(Throwable cause) {
        super(cause);
    }
}
