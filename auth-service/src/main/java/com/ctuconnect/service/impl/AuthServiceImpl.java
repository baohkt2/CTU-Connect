package com.ctuconnect.service.impl;

import com.ctuconnect.dto.AuthResponse;
import com.ctuconnect.dto.LoginRequest;
import com.ctuconnect.dto.RegisterRequest;
import com.ctuconnect.entity.EmailVerificationEntity;
import com.ctuconnect.entity.RefreshTokenEntity;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.repository.EmailVerificationRepository;
import com.ctuconnect.repository.RefreshTokenRepository;
import com.ctuconnect.repository.UserRepository;
import com.ctuconnect.security.JwtService;
import com.ctuconnect.service.AuthService;
import com.ctuconnect.service.EmailService;
import lombok.RequiredArgsConstructor;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.AuthorityUtils;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
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
        // Check if email already exists
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new RuntimeException("Email already registered");
        }

        // Create new user
        UserEntity user = UserEntity.builder()
                .email(request.getEmail())
                .username(request.getUsername())
                .password(passwordEncoder.encode(request.getPassword()))
                .role(request.getRole() != null ? request.getRole() : "USER")
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

        // Publish user created event to Kafka
        Map<String, Object> userCreatedEvent = new HashMap<>();
        userCreatedEvent.put("userId", user.getId());
        userCreatedEvent.put("email", user.getEmail());
        userCreatedEvent.put("username", user.getUsername());
        userCreatedEvent.put("role", user.getRole());
        kafkaTemplate.send("user-registration", user.getId(), userCreatedEvent);

        // Generate tokens
        String jwtToken = jwtService.generateToken(
                Map.of("role", user.getRole()),
                new org.springframework.security.core.userdetails.User(
                        user.getEmail(),
                        user.getPassword(),
                        user.isActive(),
                        true,
                        true,
                        true,
                        org.springframework.security.core.authority.AuthorityUtils.createAuthorityList("ROLE_" + user.getRole())
                )
        );

        String refreshToken = createRefreshToken(user);

        return AuthResponse.builder()
                .accessToken(jwtToken)
                .refreshToken(refreshToken)
                .email(user.getEmail())
                .username(user.getUsername())
                .role(user.getRole())
                .build();
    }

    @Override
    @Transactional
    public AuthResponse login(LoginRequest request) {
        // Authenticate user
        authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        request.getEmail(),
                        request.getPassword()
                )
        );

        // Get user
        UserEntity user = userRepository.findByEmail(request.getEmail())
                .orElseThrow(() -> new RuntimeException("User not found"));

        // Check if email is verified
        boolean isVerified = emailVerificationRepository.findByUser(user)
                .map(EmailVerificationEntity::isVerified)
                .orElse(false);

        if (!isVerified) {
            throw new RuntimeException("Email not verified. Please verify your email first.");
        }

        // Generate tokens
        UserDetails userDetails = new User(
                user.getEmail(),
                user.getPassword(),
                user.isActive(),
                true,
                true,
                true,
                AuthorityUtils.createAuthorityList("ROLE_" + user.getRole())
        );

        String jwtToken = jwtService.generateToken(Map.of("role", user.getRole()), userDetails);
        String refreshToken = createRefreshToken(user);

        return AuthResponse.builder()
                .accessToken(jwtToken)
                .refreshToken(refreshToken)
                .email(user.getEmail())
                .username(user.getUsername())
                .role(user.getRole())
                .tokenType("Bearer")
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
        UserDetails userDetails = new org.springframework.security.core.userdetails.User(
                user.getEmail(),
                user.getPassword(),
                user.isActive(),
                true,
                true,
                true,
                org.springframework.security.core.authority.AuthorityUtils.createAuthorityList("ROLE_" + user.getRole())
        );

        String jwtToken = jwtService.generateToken(Map.of("role", user.getRole()), userDetails);

        // Generate new refresh token
        refreshTokenRepository.delete(token);
        String newRefreshToken = createRefreshToken(user);

        return AuthResponse.builder()
                .accessToken(jwtToken)
                .refreshToken(newRefreshToken)
                .email(user.getEmail())
                .username(user.getUsername())
                .role(user.getRole())
                .build();
    }

    @Override
    @Transactional
    public void forgotPassword(String email) {
        UserEntity user = userRepository.findByEmail(email)
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
    public void logout(String refreshToken) {
        // Find and delete refresh token
        RefreshTokenEntity token = refreshTokenRepository.findByToken(refreshToken)
                .orElseThrow(() -> new RuntimeException("Invalid refresh token"));

        refreshTokenRepository.delete(token);
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
}
