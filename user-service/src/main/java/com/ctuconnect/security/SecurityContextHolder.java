package com.ctuconnect.security;

/**
 * Security context to hold authenticated user information throughout the request lifecycle
 */
public class SecurityContextHolder {
    private static final ThreadLocal<AuthenticatedUser> context = new ThreadLocal<>();

    public static void setAuthenticatedUser(AuthenticatedUser user) {
        context.set(user);
    }

    public static AuthenticatedUser getAuthenticatedUser() {
        return context.get();
    }

    public static void clear() {
        context.remove();
    }

    public static String getCurrentUserId() {
        AuthenticatedUser user = getAuthenticatedUser();
        return user != null ? user.getUserId() : null;
    }

    public static String getCurrentUserRole() {
        AuthenticatedUser user = getAuthenticatedUser();
        return user != null ? user.getRole() : null;
    }

    public static boolean isCurrentUserAdmin() {
        AuthenticatedUser user = getAuthenticatedUser();
        return user != null && user.isAdmin();
    }
}
