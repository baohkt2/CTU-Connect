package com.ctuconnect.init;

import com.ctuconnect.entity.EmailVerificationEntity;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.repository.EmailVerificationRepository;
import com.ctuconnect.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@Component // Cho phép Spring quét và khởi tạo class này
@RequiredArgsConstructor
public class Init implements CommandLineRunner {
    private final UserRepository userRepository;
    private final EmailVerificationRepository emailVerificationRepository;
    private final PasswordEncoder passwordEncoder;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    @Override
    public void run(String... args) {
        createDefaultAdminAccount();
    }

    private void createDefaultAdminAccount() {
        String adminEmail = "nbaocs13@gmail.com";
        String adminUsername = "Admin";
        String adminPassword = "Admin123@";

        // Kiểm tra xem admin đã tồn tại chưa
        if (userRepository.existsByEmail(adminEmail) || userRepository.existsByUsername(adminUsername)) {
            System.out.println("Admin account already exists. Skipping creation.");
            return;
        }

        // Tạo UserEntity
        UserEntity adminUser = UserEntity.builder()
                .email(adminEmail)
                .username(adminUsername)
                .password(passwordEncoder.encode(adminPassword))
                .role("ADMIN")
                .isActive(true)
                .build();

        // Lưu user vào database
        UserEntity savedUser = userRepository.save(adminUser);
        System.out.println("Created admin user with ID: " + savedUser.getId());

        // Tạo EmailVerificationEntity với trạng thái đã xác thực
        EmailVerificationEntity emailVerification = EmailVerificationEntity.builder()
                .token("admin-verified-token-" + System.currentTimeMillis())
                .user(savedUser)
                .expiryDate(System.currentTimeMillis() + 86400000L) // 24 giờ từ bây giờ
                .isVerified(true) // Đặt là đã xác thực
                .createdAt(LocalDateTime.now())
                .build();

        // Lưu email verification
        emailVerificationRepository.save(emailVerification);
        System.out.println("Created email verification for admin user with verified status: true");

        Map<String, Object> userCreatedEvent = new HashMap<>();
        userCreatedEvent.put("userId", adminUser.getId());
        userCreatedEvent.put("email", adminUser.getEmail());
        userCreatedEvent.put("username", adminUser.getUsername());
        userCreatedEvent.put("role", "LECTURER"); // Giả sử role là FACULTY
        kafkaTemplate.send("user-registration", adminUser.getId().toString(), userCreatedEvent);


        System.out.println("Default admin account created successfully!");
        System.out.println("Email: " + adminEmail);
        System.out.println("Username: " + adminUsername);
        System.out.println("Password: " + adminPassword);
        System.out.println("Role: ADMIN");
        System.out.println("Email Status: VERIFIED");
    }
}
