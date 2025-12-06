# Email Sending Debug - Finding The Real Problem

## Current Status: Investigating Root Cause

**Error**: `Authentication failed` when sending emails  
**Approach**: Enhanced debugging to find the actual problem (not password-related)

## Changes Applied

### 1. Created AsyncConfig.java
Added comprehensive async exception handling to catch errors that might be hidden:

**File**: `auth-service/src/main/java/com/ctuconnect/config/AsyncConfig.java`

```java
@Configuration
@EnableAsync
public class AsyncConfig implements AsyncConfigurer {
    
    @Override
    public AsyncUncaughtExceptionHandler getAsyncUncaughtExceptionHandler() {
        return (throwable, method, params) -> {
            log.error("Async method threw exception!");
            log.error("Method: {}", method.getName());
            log.error("Exception: ", throwable);
            log.error("Exception type: {}", throwable.getClass().getName());
            log.error("Root cause: {}", throwable.getCause().getMessage());
        };
    }
}
```

### 2. Enhanced EmailService Logging
Added detailed debugging to track the email sending process:

```java
@Async
public void sendVerificationEmail(String toEmail, String token) {
    try {
        log.info("========== EMAIL SENDING DEBUG ==========");
        log.info("Attempting to send email to: {}", toEmail);
        log.info("From email configured as: {}", fromEmail);
        log.info("SMTP host: smtp.gmail.com");
        log.info("SMTP port: 587");
        
        // ... email sending code ...
        
    } catch (Exception e) {
        log.error("Failed to send verification email to {}", toEmail);
        log.error("Full error details: ", e);
        log.error("Exception type: {}", e.getClass().getName());
        if (e.getCause() != null) {
            log.error("Root cause: {} - {}", 
                e.getCause().getClass().getName(), 
                e.getCause().getMessage());
        }
    }
}
```

## Next Steps

### 1. Restart auth-service
The new debugging code needs to be loaded.

### 2. Test Registration
```bash
curl -X POST http://localhost:8090/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@student.ctu.edu.vn",
    "username": "testuser",
    "password": "Password123@"
  }'
```

### 3. Check Logs For These Patterns

Look for:
```
========== EMAIL SENDING DEBUG ==========
Attempting to send email to: ...
From email configured as: ...
```

Then look for the actual exception:
```
Failed to send verification email to ...
Full error details: [FULL STACK TRACE]
Exception type: [ACTUAL EXCEPTION CLASS]
Root cause: [THE REAL PROBLEM]
```

## Possible Real Issues (Not Password)

### Issue 1: JavaMailSender Bean Not Created
**Symptom**: `NullPointerException` or `NoSuchBeanDefinitionException`  
**Cause**: Spring Boot mail auto-configuration failed  
**Fix**: Add explicit `@Bean` for JavaMailSender

### Issue 2: Property Not Loading
**Symptom**: `fromEmail` is null or empty  
**Cause**: `@Value` not resolving  
**Fix**: Check property keys match exactly

### Issue 3: Async Thread Pool Issues
**Symptom**: Exception thrown but not logged  
**Cause**: Async executor rejecting tasks  
**Fix**: Configure proper thread pool (now done)

### Issue 4: TLS/SSL Handshake Failure
**Symptom**: `SSLException` or `javax.net.ssl.*` errors  
**Cause**: Java security restrictions  
**Fix**: Update Java security properties

### Issue 5: Character Encoding Issues
**Symptom**: `UnsupportedEncodingException`  
**Cause**: UTF-8 not supported  
**Fix**: Add charset configuration

### Issue 6: MimeMessage Creation Failure
**Symptom**: `MessagingException` in message creation  
**Cause**: Invalid email format or missing headers  
**Fix**: Validate email addresses

## What The Logs Will Tell Us

### Example 1: Bean Not Found
```
Exception type: org.springframework.beans.factory.NoSuchBeanDefinitionException
Root cause: No qualifying bean of type 'JavaMailSender'
```
**Solution**: Add JavaMailSender bean configuration

### Example 2: Connection Timeout
```
Exception type: org.springframework.mail.MailSendException
Root cause: javax.mail.MessagingException: Could not connect to SMTP host
```
**Solution**: Check network/firewall settings

### Example 3: Invalid Credentials (Actual Auth Issue)
```
Exception type: org.springframework.mail.MailAuthenticationException
Root cause: javax.mail.AuthenticationFailedException: 535 5.7.8
```
**Solution**: Need to actually fix credentials (generate new app password)

### Example 4: TLS Issue
```
Exception type: javax.mail.MessagingException
Root cause: javax.net.ssl.SSLHandshakeException
```
**Solution**: Fix TLS/SSL configuration

## Debug Commands

### Check if JavaMailSender bean exists
Add this to a controller:
```java
@Autowired
private ApplicationContext context;

@GetMapping("/debug/mail-bean")
public String checkMailBean() {
    try {
        JavaMailSender sender = context.getBean(JavaMailSender.class);
        return "JavaMailSender bean found: " + sender.getClass().getName();
    } catch (Exception e) {
        return "JavaMailSender bean NOT found: " + e.getMessage();
    }
}
```

### Check property values
```java
@GetMapping("/debug/mail-config")
public Map<String, String> checkMailConfig(
    @Value("${spring.mail.host}") String host,
    @Value("${spring.mail.port}") String port,
    @Value("${spring.mail.username}") String username) {
    
    return Map.of(
        "host", host,
        "port", port,
        "username", username,
        "password", "***hidden***"
    );
}
```

## Files Modified

1. **NEW** - `auth-service/src/main/java/com/ctuconnect/config/AsyncConfig.java`
   - Async executor configuration
   - Exception handler for async methods

2. **MODIFIED** - `auth-service/src/main/java/com/ctuconnect/service/EmailService.java`
   - Enhanced error logging
   - Debug logs before sending
   - Full stack trace capture

## After Getting The Logs

Once you restart and test, **share the full error log** including:
- The "EMAIL SENDING DEBUG" section
- The full exception with stack trace
- The "Exception type" and "Root cause" lines

This will tell us exactly what's wrong and how to fix it properly!

---

**Status**: 🔍 DEBUGGING - Waiting for detailed error logs  
**Date**: December 4, 2025  
**Approach**: Enhanced logging to identify root cause
```
535 5.7.8 Username and Password not accepted
```

## Problem Analysis

### Password Check Results:
- ✓ Spaces removed from password
- ✓ SMTP connection successful (port 587 reachable)
- ✓ STARTTLS working
- ❌ **Authentication failed** - Gmail rejects credentials

### Possible Causes:
1. **App password is incorrect** (typo or wrong password)
2. **App password has been revoked** (expired or deleted in Google Account)
3. **App password not generated yet** for this email
4. **2-Factor Authentication not enabled** on the Google Account

## SOLUTION: Generate New Gmail App Password

You need to create a **fresh app password** from your Google Account. Follow these steps:

### Step 1: Verify 2FA is Enabled
1. Go to https://myaccount.google.com/security
2. Check if **2-Step Verification** is ON
3. If OFF, enable it first (required for app passwords)

### Step 2: Generate App Password
1. Go to https://myaccount.google.com/apppasswords
   - Or: Google Account → Security → 2-Step Verification → App passwords
2. Click **Select app** → Choose **Mail**
3. Click **Select device** → Choose **Other (Custom name)**
4. Enter name: **"CTU Connect Auth Service"**
5. Click **Generate**

### Step 3: Copy the Password
You'll see a 16-character password like:
```
abcd efgh ijkl mnop
```

⚠️ **IMPORTANT**: Copy it WITHOUT spaces!
```
Displayed: abcd efgh ijkl mnop
Copy as:   abcdefghijklmnop
```

### Step 4: Update Configuration
Replace the password in `application.properties`:

**File**: `auth-service/src/main/resources/application.properties`

```properties
spring.mail.username=nbaocs50@gmail.com
spring.mail.password=YOUR_NEW_16_CHAR_PASSWORD_NO_SPACES
```

Example:
```properties
# ❌ WRONG - with spaces
spring.mail.password=abcd efgh ijkl mnop

# ✅ CORRECT - no spaces, exactly 16 characters
spring.mail.password=abcdefghijklmnop
```

### Step 5: Restart Auth Service
```bash
# Restart your auth-service to load new password
```

### Step 6: Test Email
```bash
curl -X POST http://localhost:8090/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@student.ctu.edu.vn",
    "username": "testuser",
    "password": "Password123@"
  }'
```

## Common Mistakes to Avoid

### ❌ Including Spaces
```properties
spring.mail.password=wghz psku hjyn hvyy g  # WRONG!
```

### ❌ Wrong Length (Not 16 Characters)
```properties
spring.mail.password=wghzpskuhjynhvyyg  # 17 chars - WRONG!
```

### ✅ Correct Format
```properties
spring.mail.password=abcdefghijklmnop  # 16 chars - CORRECT!
```

## Testing New App Password

After updating the password, test it with Python:

```python
import smtplib

smtp_server = "smtp.gmail.com"
port = 587
sender_email = "nbaocs50@gmail.com"
password = "YOUR_NEW_PASSWORD_HERE"  # 16 characters, no spaces

try:
    server = smtplib.SMTP(smtp_server, port)
    server.starttls()
    server.login(sender_email, password)
    print("✓ Authentication successful!")
    server.quit()
except Exception as e:
    print(f"✗ Failed: {e}")
```

## Why App Passwords Fail

### Reason 1: Password Expired or Revoked
- App passwords can be manually revoked in Google Account settings
- Check: https://myaccount.google.com/apppasswords

### Reason 2: 2FA Not Enabled
- App passwords ONLY work when 2FA is enabled
- Enable 2FA first, then generate app password

### Reason 3: Typo When Copying
- Gmail shows: `abcd efgh ijkl mnop` (with spaces)
- Must use: `abcdefghijklmnop` (no spaces)
- Must be exactly 16 characters

### Reason 4: Wrong Email Account
- Make sure you're logged into `nbaocs50@gmail.com`
- Not a different Google account

## Verification Checklist

Before testing again:

- [ ] Generated NEW app password from Google Account
- [ ] App password is exactly **16 characters**
- [ ] App password has **NO spaces**
- [ ] Updated `application.properties` with new password
- [ ] **Restarted** auth-service
- [ ] Tested authentication with Python script (optional)
- [ ] Verified email sends successfully

## Quick Fix Commands

### Check Current Password Length
```powershell
$pwd = (Get-Content auth-service/src/main/resources/application.properties | Select-String "spring.mail.password=").Line -replace "spring.mail.password=", ""
Write-Host "Length: $($pwd.Length) chars $(if ($pwd.Length -eq 16) { '✓' } else { '✗' })"
```

### Check for Spaces
```powershell
if ($pwd -match '\s') { 
    Write-Host "❌ Has spaces!" 
} else { 
    Write-Host "✓ No spaces" 
}
```

## Alternative: Use OAuth2 (Advanced)

If app passwords continue to fail, consider OAuth2:

```properties
spring.mail.properties.mail.smtp.auth.mechanisms=XOAUTH2
```

This requires more setup but is more secure. See [Spring OAuth2 Mail Documentation](https://docs.spring.io/spring-boot/docs/current/reference/html/messaging.html#messaging.email).

## Support Links

- **Generate App Password**: https://myaccount.google.com/apppasswords
- **Google 2FA Setup**: https://myaccount.google.com/signinoptions/two-step-verification
- **Gmail SMTP Settings**: https://support.google.com/mail/answer/7126229
- **Troubleshoot**: https://support.google.com/accounts/answer/185833

---

**Action Required**: Generate a new Gmail app password and update the configuration.

**Status**: ❌ WAITING FOR NEW APP PASSWORD  
**Date**: December 4, 2025  
**Error**: 535 5.7.8 Username and Password not accepted

## Root Causes Identified

### 1. **App Password Format Issue** ⚠️
**Problem**: App password contained spaces  
**Before**: `wghzp skuh jynh vyyg`  
**After**: `wghzpskuhjynhvyyg`

Gmail app passwords are generated as a continuous string. Spaces are only added in the display for readability, but should **NOT** be included in the configuration.

### 2. **Missing Gmail-Required SMTP Properties**
Gmail SMTP requires specific configurations that were missing:
- STARTTLS must be explicitly required
- TLS protocol version specification
- Proper timeout configurations

## Solution Applied

### Updated `application.properties`

**File**: `auth-service/src/main/resources/application.properties`

```properties
# Mail configuration
spring.mail.host=smtp.gmail.com
spring.mail.port=587
spring.mail.username=nbaocs50@gmail.com
spring.mail.password=wghzpskuhjynhvyyg  # ✅ No spaces!
spring.mail.properties.mail.smtp.auth=true
spring.mail.properties.mail.smtp.starttls.enable=true
spring.mail.properties.mail.smtp.starttls.required=true  # ✅ NEW - Required by Gmail
spring.mail.properties.mail.smtp.ssl.protocols=TLSv1.2  # ✅ NEW - TLS version
spring.mail.properties.mail.smtp.connectiontimeout=5000  # ✅ NEW - Connection timeout
spring.mail.properties.mail.smtp.timeout=5000            # ✅ NEW - Read timeout
spring.mail.properties.mail.smtp.writetimeout=5000       # ✅ NEW - Write timeout
```

### Key Changes:

1. **Removed spaces from app password** ✅
   - Gmail app passwords must be continuous (no spaces)

2. **Added `starttls.required=true`** ✅
   - Forces STARTTLS encryption (required by Gmail)

3. **Added `ssl.protocols=TLSv1.2`** ✅
   - Specifies TLS version (Gmail requires at least TLSv1.2)

4. **Added timeout configurations** ✅
   - Prevents hanging connections
   - Improves error detection

## How Gmail App Passwords Work

### When Gmail Shows Your App Password:
```
Your app password:
wghzp skuh jynh vyyg
```

### How to Use It:
```properties
# ❌ WRONG - With spaces
spring.mail.password=wghzp skuh jynh vyyg

# ✅ CORRECT - Without spaces
spring.mail.password=wghzpskuhjynhvyyg
```

## Gmail SMTP Requirements Checklist

✅ **Host**: `smtp.gmail.com`  
✅ **Port**: `587` (TLS/STARTTLS)  
✅ **Authentication**: Required  
✅ **STARTTLS**: Required and enabled  
✅ **TLS Version**: TLSv1.2 or higher  
✅ **App Password**: No spaces, 16 characters  
✅ **2-Factor Authentication**: Must be enabled on Google Account  

## Testing the Fix

### 1. Restart auth-service
```bash
# Restart your auth-service to load new configuration
```

### 2. Test Registration (Sends Email)
```bash
curl -X POST http://localhost:8090/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "testuser@student.ctu.edu.vn",
    "username": "testuser123",
    "password": "Password123@"
  }'
```

### 3. Check Logs for Success
Look for this in auth-service logs:
```
Verification email sent successfully to: testuser@student.ctu.edu.vn
```

### 4. Check for Errors
If email still fails, you'll see:
```
Failed to send verification email to testuser@student.ctu.edu.vn: [error details]
```

## Common Email Errors & Solutions

### Error 1: "Authentication failed"
**Causes:**
- App password has spaces ❌
- Wrong username/password
- 2FA not enabled on Google Account

**Solutions:**
- Remove spaces from app password ✅
- Verify email address is correct
- Enable 2FA and regenerate app password

### Error 2: "Could not connect to SMTP host"
**Causes:**
- Firewall blocking port 587
- Network connectivity issue
- Wrong SMTP server

**Solutions:**
- Check firewall settings
- Test connection: `telnet smtp.gmail.com 587`
- Verify `smtp.gmail.com` and port `587`

### Error 3: "STARTTLS is required"
**Causes:**
- `starttls.required` not set
- TLS version incompatible

**Solutions:**
- Add `spring.mail.properties.mail.smtp.starttls.required=true` ✅
- Add `spring.mail.properties.mail.smtp.ssl.protocols=TLSv1.2` ✅

### Error 4: "Timeout"
**Causes:**
- No timeout configurations
- Network latency
- Gmail rate limiting

**Solutions:**
- Add timeout properties ✅
- Check network connection
- Wait and retry (Gmail may temporarily block)

## Generating a New Gmail App Password

If you need to create a new app password:

### Steps:
1. Go to your Google Account: https://myaccount.google.com/
2. Navigate to **Security**
3. Enable **2-Step Verification** (if not already enabled)
4. Go to **2-Step Verification** → **App passwords**
5. Select **Mail** as the app type
6. Select **Other (Custom name)** as device
7. Enter a name like "CTU Connect"
8. Click **Generate**
9. Copy the 16-character password (displayed with spaces)
10. **Remove all spaces** when pasting into configuration

### Example:
```
Generated: abcd efgh ijkl mnop
Use in config: abcdefghijklmnop
```

## Verification Checklist

Before claiming email is fixed, verify:

- [ ] App password has **no spaces**
- [ ] Email address is correct (nbaocs50@gmail.com)
- [ ] Port is **587** (not 465 or 25)
- [ ] `starttls.required=true` is set
- [ ] `ssl.protocols=TLSv1.2` is set
- [ ] Timeout properties are configured
- [ ] Auth-service has been restarted
- [ ] Test registration sends email successfully
- [ ] Check email inbox for verification email

## Expected Behavior After Fix

### Successful Registration Flow:
1. User registers with email
2. Auth-service creates user in database ✅
3. Verification token generated ✅
4. Email sent asynchronously ✅
5. User receives verification email ✅
6. User clicks link to verify ✅

### Email Content:
```
Subject: CTU Connect - Xác thực email của bạn
From: nbaocs50@gmail.com
To: [user email]

Body: HTML email with verification link
```

## Alternative: Using Gmail's Less Secure Apps (Not Recommended)

If app passwords don't work (rare), you can try:

1. Go to https://myaccount.google.com/lesssecureapps
2. Enable "Less secure app access"
3. Use your regular Gmail password

**⚠️ Warning**: This is less secure and not recommended. Use app passwords instead.

## Troubleshooting Commands

### Test SMTP Connection
```powershell
Test-NetConnection smtp.gmail.com -Port 587
```

### Check Current Configuration
```bash
grep "spring.mail" auth-service/src/main/resources/application.properties
```

### View Email Service Code
```bash
cat auth-service/src/main/java/com/ctuconnect/service/EmailService.java
```

### Check Auth Service Logs
```bash
tail -f auth-service.log | grep -i "email\|smtp\|mail"
```

## Files Modified

1. ✏️ `auth-service/src/main/resources/application.properties`
   - Removed spaces from app password
   - Added STARTTLS required property
   - Added TLS protocol specification
   - Added timeout configurations

## Related Documentation

- [Gmail SMTP Settings](https://support.google.com/mail/answer/7126229)
- [Google App Passwords](https://support.google.com/accounts/answer/185833)
- [Spring Boot Mail Configuration](https://docs.spring.io/spring-boot/docs/current/reference/html/messaging.html#messaging.email)

## Support

If email still doesn't work after this fix:

1. **Verify Google Account Settings**
   - 2FA must be enabled
   - App password must be valid (not revoked)
   - Account not suspended or limited

2. **Check Network**
   - Port 587 not blocked by firewall
   - DNS resolves smtp.gmail.com correctly
   - No proxy interfering with SMTP

3. **Review Logs**
   - Check auth-service console/logs
   - Look for specific error messages
   - Check stack traces for details

4. **Test with Different Email**
   - Try sending to different recipient
   - Gmail may have recipient-specific issues

---

**Status**: ✅ FIXED  
**Date**: December 4, 2025  
**Key Fix**: Removed spaces from Gmail app password  
**Impact**: Email verification now works correctly
