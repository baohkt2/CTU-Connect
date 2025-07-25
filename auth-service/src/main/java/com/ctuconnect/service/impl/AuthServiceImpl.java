package com.ctuconnect.service.impl;

import com.ctuconnect.dto.AuthResponse;
import com.ctuconnect.dto.LoginRequest;
import com.ctuconnect.dto.RegisterRequest;
import com.ctuconnect.entity.EmailVerificationEntity;
import com.ctuconnect.entity.RefreshTokenEntity;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.exception.EmailAlreadyExistsException;
import com.ctuconnect.exception.UsernameAlreadyExistsException;
import com.ctuconnect.mapper.UserMapper;
import com.ctuconnect.repository.EmailVerificationRepository;
import com.ctuconnect.repository.RefreshTokenRepository;
import com.ctuconnect.repository.UserRepository;
import com.ctuconnect.security.CustomUserPrincipal;
import com.ctuconnect.security.JwtService;
import com.ctuconnect.service.AuthService;
import com.ctuconnect.service.EmailService;
import lombok.RequiredArgsConstructor;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class AuthServiceImpl implements AuthService {

    private final UserRepository userRepository;
    private final RefreshTokenRepository refreshTokenRepository;
    private final EmailVerificationRepository emailVerificationRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;
    private final AuthenticationManager authenticationManager;
    private final EmailService emailService;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    @Override
    @Transactional
    public AuthResponse register(RegisterRequest request) {
        // Normalize email and username to lowercase
        String normalizedEmail = request.getEmail().toLowerCase().trim();
        String normalizedUsername = request.getUsername().toLowerCase().trim();

        // Check if email already exists (case-insensitive)
        if (userRepository.existsByEmail(normalizedEmail)) {
            throw new EmailAlreadyExistsException("Email đã được đăng ký. Vui lòng sử dụng email khác.");
        }

        // Check if username already exists (case-insensitive)
        if (userRepository.existsByUsername(normalizedUsername)) {
            throw new UsernameAlreadyExistsException("Tên đăng nhập đã được sử dụng. Vui lòng chọn tên khác.");
        }

        // Determine role for user-service based on email domain
        String userServiceRole = determineUserServiceRole(normalizedEmail);

        // Create new user with normalized email and username - always save as "USER" in auth-db
        UserEntity user = UserEntity.builder()
                .email(normalizedEmail)
                .username(normalizedUsername)
                .password(passwordEncoder.encode(request.getPassword()))
                .role("USER") // Always save as "USER" in auth-db
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .isActive(true)
                .build();

        userRepository.save(user);

        // Create email verification token
        String verificationToken = UUID.randomUUID().toString();
        EmailVerificationEntity verification = EmailVerificationEntity.builder()
                .token(verificationToken)
                .user(user)
                .expiryDate(Instant.now().plus(24, ChronoUnit.HOURS).toEpochMilli())
                .isVerified(false)
                .createdAt(LocalDateTime.now())
                .build();

        emailVerificationRepository.save(verification);

        // Send verification email
        emailService.sendVerificationEmail(user.getEmail(), verificationToken);

        // Publish user created event to Kafka with role based on email domain
        Map<String, Object> userCreatedEvent = new HashMap<>();
        userCreatedEvent.put("userId", user.getId());
        userCreatedEvent.put("email", user.getEmail());
        userCreatedEvent.put("username", user.getUsername());
        userCreatedEvent.put("role", userServiceRole); // Use determined role for user-service
        kafkaTemplate.send("user-registration", user.getId().toString(), userCreatedEvent);

        // Generate tokens
        CustomUserPrincipal userPrincipal = new CustomUserPrincipal(
                user.getId().toString(), // Convert Long to String for JWT
                user.getEmail(),
                user.getPassword(),
                user.getRole(),
                user.getIsActive(),
                List.of(new org.springframework.security.core.authority.SimpleGrantedAuthority("ROLE_" + user.getRole()))
        );

        String jwtToken = jwtService.generateToken(userPrincipal);
        String refreshToken = createRefreshToken(user);

        return AuthResponse.builder()
                .accessToken(jwtToken)
                .refreshToken(refreshToken)
                .tokenType("Bearer")
                .user(UserMapper.toDto(user))
                .build();

    }

    @Override
    @Transactional
    public AuthResponse login(LoginRequest request) {
        // Normalize identifier to lowercase for search
        String normalizedIdentifier = request.getIdentifier().toLowerCase().trim();

        // Tìm user bằng email hoặc username (case-insensitive)
        UserEntity user = userRepository.findByEmailOrUsername(normalizedIdentifier)
                .orElseThrow(() -> new RuntimeException("User not found"));

        // Authenticate user - sử dụng email làm username principal
        authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        user.getEmail(), // Luôn sử dụng email làm username principal
                        request.getPassword()
                )
        );

        // Check if email is verified
        boolean isVerified = emailVerificationRepository.findByUser(user)
                .map(EmailVerificationEntity::isVerified)
                .orElse(false);

        if (!isVerified) {
            throw new RuntimeException("Email not verified. Please verify your email first.");
        }

        // Generate tokens
        CustomUserPrincipal userPrincipal = new CustomUserPrincipal(
                user.getId().toString(), // Convert Long to String for JWT
                user.getEmail(),
                user.getPassword(),
                user.getRole(),
                user.isActive(),
                List.of(new org.springframework.security.core.authority.SimpleGrantedAuthority("ROLE_" + user.getRole()))
        );

        String jwtToken = jwtService.generateToken(userPrincipal);
        String refreshToken = createRefreshToken(user);

        return AuthResponse.builder()
                .accessToken(jwtToken)
                .refreshToken(refreshToken)
                .tokenType("Bearer")
                .user(UserMapper.toDto(user))
                .build();


    }

    @Override
    @Transactional
    public AuthResponse refreshToken(String refreshToken) {
        // Validate refresh token
        RefreshTokenEntity token = refreshTokenRepository.findByToken(refreshToken)
                .orElseThrow(() -> new RuntimeException("Invalid refresh token"));

        if (token.getExpiryDate().isBefore(LocalDateTime.now())) {
            refreshTokenRepository.delete(token);
            throw new RuntimeException("Refresh token expired");
        }

        UserEntity user = token.getUser();

        // Generate new access token
        CustomUserPrincipal userPrincipal = new CustomUserPrincipal(
                user.getId().toString(), // Convert Long to String for JWT
                user.getEmail(),
                user.getPassword(),
                user.getRole(),
                user.isActive(),
                List.of(new org.springframework.security.core.authority.SimpleGrantedAuthority("ROLE_" + user.getRole()))
        );

        String jwtToken = jwtService.generateToken(userPrincipal);

        // Generate new refresh token
        refreshTokenRepository.delete(token);
        String newRefreshToken = createRefreshToken(user);

        return AuthResponse.builder()
                .accessToken(jwtToken)
                .refreshToken(refreshToken)
                .tokenType("Bearer")
                .user(UserMapper.toDto(user))
                .build();

    }

    @Override
    @Transactional
    public void forgotPassword(String email) {
        // Normalize email to lowercase
        String normalizedEmail = email.toLowerCase().trim();

        UserEntity user = userRepository.findByEmail(normalizedEmail)
                .orElseThrow(() -> new RuntimeException("User not found"));

        // Generate password reset token
        String resetToken = UUID.randomUUID().toString();

        // Save token to database
        // Note: In a real implementation, you would save this token to a dedicated table
        // Here we're reusing the EmailVerificationEntity for simplicity
        EmailVerificationEntity resetTokenEntity = emailVerificationRepository.findByUser(user)
                .orElse(new EmailVerificationEntity());

        resetTokenEntity.setUser(user);
        resetTokenEntity.setToken(resetToken);
        resetTokenEntity.setExpiryDate(Instant.now().plus(1, ChronoUnit.HOURS).toEpochMilli());
        resetTokenEntity.setCreatedAt(LocalDateTime.now());

        emailVerificationRepository.save(resetTokenEntity);

        // Send password reset email
        emailService.sendPasswordResetEmail(email, resetToken);

        // Publish password reset event
        Map<String, Object> passwordResetEvent = new HashMap<>();
        passwordResetEvent.put("userId", user.getId());
        passwordResetEvent.put("email", user.getEmail());
        passwordResetEvent.put("resetToken", resetToken);
        passwordResetEvent.put("eventType", "REQUESTED");
        kafkaTemplate.send("password-reset", user.getId().toString(), passwordResetEvent);
    }

    @Override
    @Transactional
    public void resetPassword(String token, String newPassword) {
        // Find token
        EmailVerificationEntity resetToken = emailVerificationRepository.findByToken(token)
                .orElseThrow(() -> new RuntimeException("Invalid or expired token"));

        // Check if token is expired
        if (resetToken.getExpiryDate() < Instant.now().toEpochMilli()) {
            throw new RuntimeException("Token expired");
        }

        // Update password
        UserEntity user = resetToken.getUser();
        user.setPassword(passwordEncoder.encode(newPassword));
        user.setUpdatedAt(LocalDateTime.now());

        userRepository.save(user);

        // Delete token
        resetToken.setExpiryDate(0L); // Invalidate token
        emailVerificationRepository.save(resetToken);

        // Publish password reset completed event
        Map<String, Object> passwordResetEvent = new HashMap<>();
        passwordResetEvent.put("userId", user.getId());
        passwordResetEvent.put("email", user.getEmail());
        passwordResetEvent.put("resetToken", token);
        passwordResetEvent.put("eventType", "COMPLETED");
        kafkaTemplate.send("password-reset", user.getId().toString(), passwordResetEvent);

        // Publish user updated event
        Map<String, Object> userUpdatedEvent = new HashMap<>();
        userUpdatedEvent.put("userId", user.getId());
        userUpdatedEvent.put("email", user.getEmail());
        userUpdatedEvent.put("username", user.getUsername());
        userUpdatedEvent.put("role", user.getRole());
        userUpdatedEvent.put("isActive", user.isActive());
        kafkaTemplate.send("user-updated", user.getId().toString(), userUpdatedEvent);
    }

    @Override
    @Transactional
    public void verifyEmail(String token) {
        // Find token
        EmailVerificationEntity verification = emailVerificationRepository.findByToken(token)
                .orElseThrow(() -> new RuntimeException("Invalid verification token"));

        // Check if token is expired
        if (verification.getExpiryDate() < Instant.now().toEpochMilli()) {
            throw new RuntimeException("Verification token expired");
        }

        // Mark email as verified
        verification.setVerified(true);
        emailVerificationRepository.save(verification);

        // Publish user verification event
        UserEntity user = verification.getUser();
        Map<String, Object> userVerificationEvent = new HashMap<>();
        userVerificationEvent.put("userId", user.getId());
        userVerificationEvent.put("email", user.getEmail());
        userVerificationEvent.put("verificationToken", token);
        userVerificationEvent.put("isVerified", true);
        kafkaTemplate.send("user-verification", user.getId().toString(), userVerificationEvent);
    }

    @Override
    @Transactional
    public void resendVerificationEmail(String oldToken) {
        // Find the old verification token
        EmailVerificationEntity oldVerification = emailVerificationRepository.findByToken(oldToken)
                .orElseThrow(() -> new RuntimeException("Invalid verification token"));

        UserEntity user = oldVerification.getUser();

        // Check if email is already verified
        if (oldVerification.isVerified()) {
            throw new RuntimeException("Email is already verified");
        }

        // Generate new verification token
        String newVerificationToken = UUID.randomUUID().toString();

        // Update the verification entity with new token and expiry
        oldVerification.setToken(newVerificationToken);
        oldVerification.setExpiryDate(Instant.now().plus(24, ChronoUnit.HOURS).toEpochMilli());
        oldVerification.setCreatedAt(LocalDateTime.now());

        emailVerificationRepository.save(oldVerification);

        // Send new verification email
        emailService.sendVerificationEmail(user.getEmail(), newVerificationToken);

        // Publish resend verification event
        Map<String, Object> resendVerificationEvent = new HashMap<>();
        resendVerificationEvent.put("userId", user.getId());
        resendVerificationEvent.put("email", user.getEmail());
        resendVerificationEvent.put("oldToken", oldToken);
        resendVerificationEvent.put("newToken", newVerificationToken);
        resendVerificationEvent.put("eventType", "RESENT");
        kafkaTemplate.send("user-verification", user.getId().toString(), resendVerificationEvent);
    }

    @Override
    @Transactional
    public void logout(String refreshToken) {
        // Find and delete refresh token
        RefreshTokenEntity token = refreshTokenRepository.findByToken(refreshToken)
                .orElseThrow(() -> new RuntimeException("Invalid refresh token"));

        refreshTokenRepository.delete(token);
    }

    @Override
    public AuthResponse getCurrentUser(String accessToken) {
        try {
            // Extract user information from access token
            String userEmail = jwtService.extractUsername(accessToken);

            // Validate token
            if (jwtService.isTokenExpired(accessToken)) {
                throw new RuntimeException("Access token has expired");
            }

            // Find user by email
            UserEntity user = userRepository.findByEmail(userEmail)
                    .orElseThrow(() -> new RuntimeException("User not found"));

            // Check if user is active
            if (!user.isActive()) {
                throw new RuntimeException("User account is inactive");
            }

            return AuthResponse.builder()
                    .user(UserMapper.toDto(user))
                    .tokenType("Bearer")
                    .build();
        } catch (Exception e) {
            throw new RuntimeException("Invalid access token: " + e.getMessage());
        }
    }

    /**
     * Helper method to create a refresh token
     */
    private String createRefreshToken(UserEntity user) {
        // Delete any existing refresh tokens for this user
        refreshTokenRepository.deleteByUser(user);

        // Create new refresh token
        RefreshTokenEntity refreshToken = RefreshTokenEntity.builder()
                .user(user)
                .token(UUID.randomUUID().toString())
                .expiryDate(LocalDateTime.now().plusDays(7))
                .createdAt(LocalDateTime.now())
                .build();

        refreshTokenRepository.save(refreshToken);

        return refreshToken.getToken();
    }

    /**
     * Helper method to determine user-service role based on email domain
     */
    private String determineUserServiceRole(String email) {
        if (email.endsWith("@student.ctu.edu.vn")) {
            return "STUDENT";
        } else if (email.endsWith("@ctu.edu.vn")) {
            return "FACULTY";
        }
        return "USER";
    }
}
