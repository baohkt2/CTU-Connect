//package com.ctuconnect.service;
//
//import com.ctuconnect.entity.RefreshToken;
//import com.ctuconnect.entity.User;
//import com.ctuconnect.repository.RefreshTokenRepository;
//import com.ctuconnect.repository.UserRepository;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.stereotype.Service;
//import org.springframework.transaction.annotation.Transactional;
//
//import java.time.Instant;
//import java.util.Optional;
//import java.util.UUID;
//
//@Service
//public class RefreshTokenService {
//
//    @Value("${jwt.refresh.expiration.ms}")
//    private long refreshTokenDurationMs;
//
//    private final RefreshTokenRepository refreshTokenRepository;
//    private final UserRepository userRepository;
//
//    public RefreshTokenService(RefreshTokenRepository refreshTokenRepository, UserRepository userRepository) {
//        this.refreshTokenRepository = refreshTokenRepository;
//        this.userRepository = userRepository;
//    }
//
//    public Optional<RefreshToken> findByToken(String token) {
//        return refreshTokenRepository.findByToken(token);
//    }
//
//    public RefreshToken createRefreshToken(User user) {
//        // Xóa bất kỳ refresh token hiện có nào của người dùng này
//        deleteRefreshTokensByUserId(user.getId()); // Gọi phương thức mới deleteByUserId
//
//        RefreshToken refreshToken = RefreshToken.builder()
//                .user(user)
//                .token(UUID.randomUUID().toString())
//                .expiryDate(Instant.now().plusMillis(refreshTokenDurationMs))
//                .build();
//
//        return refreshTokenRepository.save(refreshToken);
//    }
//
//    public RefreshToken verifyExpiration(RefreshToken token) {
//        if (token.isExpired()) {
//            refreshTokenRepository.delete(token); // Xóa token đã hết hạn
//            throw new RuntimeException("Refresh token was expired. Please make a new login request");
//        }
//        return token;
//    }
//
//    @Transactional
//    public void deleteByToken(String token) {
//        refreshTokenRepository.deleteByToken(token);
//    }
//
//    // Thêm phương thức này vào RefreshTokenService
//    @Transactional
//    public void deleteRefreshTokensByUserId(Long userId) {
//        refreshTokenRepository.deleteByUserId(userId);
//    }
//
//    // Nếu bạn muốn giữ lại phương thức deleteByUser(User user) trong service,
//    // bạn có thể làm như sau để nó gọi phương thức mới:
//    /*
//    @Transactional
//    public void deleteByUser(User user) {
//        if (user != null && user.getId() != null) {
//            refreshTokenRepository.deleteByUserId(user.getId());
//        }
//    }
//    */
//}