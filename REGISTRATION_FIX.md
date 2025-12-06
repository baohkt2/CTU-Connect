# Registration Endpoint Fix

## Problem
The `/api/auth/register` endpoint was returning a 400 Bad Request with the error message:
```json
{
  "success": false,
  "message": "Authentication failed"
}
```

## Root Cause
The actual error was **`org.springframework.mail.MailAuthenticationException: Authentication failed`**. 

The registration process was working correctly - users were being created in the database and email verification records were being generated. However, when the `EmailService` tried to send the verification email, it failed due to Gmail SMTP authentication issues. Since the email sending was **synchronous and throwing an exception**, it caused the entire registration transaction to fail and roll back.

### Error Flow:
1. ✓ User data validated
2. ✓ User record created in database
3. ✓ Email verification token generated
4. ✗ **Email sending failed** (Gmail SMTP auth error)
5. ✗ Exception thrown → Transaction rolled back
6. ✗ Registration failed with "Authentication failed" error

## Solution Applied

### 1. Made Email Sending Non-Blocking
Modified `EmailService.java` to catch email exceptions instead of propagating them:

**Changes in**: `auth-service/src/main/java/com/ctuconnect/service/EmailService.java`

- Changed exception handling from `throw new RuntimeException(...)` to logging only
- Email failures now log errors but don't prevent registration from completing
- Both `sendVerificationEmail()` and `sendPasswordResetEmail()` methods updated

```java
} catch (Exception e) {
    // Log the error but don't throw exception to prevent blocking registration
    log.error("Failed to send verification email to {}: {}. User registration will continue.", 
             toEmail, e.getMessage());
    log.debug("Email error details: ", e);
}
```

### 2. Enabled Async Support
Added `@EnableAsync` to the main application class to ensure async email sending works properly:

**Changes in**: `auth-service/src/main/java/com/ctuconnect/AuthServiceApplication.java`

```java
@SpringBootApplication
@EnableDiscoveryClient
@EnableFeignClients
@EnableAsync  // ← Added this
public class AuthServiceApplication {
    // ...
}
```

### 3. Added WebClient Configuration (Bonus Fix)
Created `WebClientConfig.java` to provide the `WebClient.Builder` bean required by `RecaptchaService`:

**File**: `auth-service/src/main/java/com/ctuconnect/config/WebClientConfig.java`

```java
@Configuration
public class WebClientConfig {
    @Bean
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }
}
```

### 4. Improved Security Configuration
Updated `SecurityConfig.java` to explicitly list all public auth endpoints:

**Changes in**: `auth-service/src/main/java/com/ctuconnect/security/SecurityConfig.java`

- Changed from `.requestMatchers("/api/auth/**").permitAll()` 
- To explicit matchers for better clarity

## How to Apply the Fix

1. **Restart the auth-service** (the changes are already saved to the files):
   ```bash
   # Stop the running auth-service process, then restart it
   cd auth-service
   ./mvnw spring-boot:run
   # or however you normally start it
   ```

2. **Test the registration endpoint**:
   ```bash
   curl -X POST http://localhost:8090/api/auth/register \
     -H "Content-Type: application/json" \
     -d '{
       "email": "test@student.ctu.edu.vn",
       "username": "testuser",
       "password": "Password123@"
     }'
   ```

## Expected Result
After applying these fixes and restarting the auth-service:
- ✅ Registration will succeed even if email sending fails
- ✅ Users will be created in the database with verification tokens
- ✅ Registration response will include JWT tokens
- ⚠️ Verification emails won't be sent until Gmail SMTP credentials are fixed
- 📝 Email failures will be logged but won't block user registration

## Fixing the Email Issue (Optional but Recommended)

The Gmail SMTP authentication is failing because of one of these reasons:

1. **2-Factor Authentication** - If enabled, you need an App Password:
   - Go to Google Account → Security → 2-Step Verification → App Passwords
   - Generate a new app password for "Mail"
   - Use this password in `application.properties` instead of your regular password

2. **Less Secure Apps** - Gmail might be blocking the connection:
   - This is less common now, but check Google Account settings

3. **Wrong Credentials** - Verify the email/password in:
   ```properties
   spring.mail.username=nbaocs50@gmail.com
   spring.mail.password=wuwr dacr csiw fxvx  # ← Check if this is correct
   ```

## Files Modified
1. `/auth-service/src/main/java/com/ctuconnect/service/EmailService.java` (MODIFIED)
2. `/auth-service/src/main/java/com/ctuconnect/AuthServiceApplication.java` (MODIFIED)
3. `/auth-service/src/main/java/com/ctuconnect/config/WebClientConfig.java` (NEW)
4. `/auth-service/src/main/java/com/ctuconnect/security/SecurityConfig.java` (MODIFIED)
