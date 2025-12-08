package vn.ctu.edu.recommend.client;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.model.dto.UserDTO;
import vn.ctu.edu.recommend.model.dto.UserAcademicProfile;

import java.util.Collections;
import java.util.List;

/**
 * Fallback implementation for User Service Feign Client
 * Returns default/empty responses when user-service is unavailable
 */
@Component
@Slf4j
public class UserServiceFeignClientFallback implements UserServiceFeignClient {

    @Override
    public UserDTO getUserById(String userId) {
        log.warn("User Service unavailable. Returning null for user: {}", userId);
        return null;
    }

    @Override
    public UserAcademicProfile getUserAcademicProfile(String userId) {
        log.warn("User Service unavailable. Returning default profile for user: {}", userId);
        return UserAcademicProfile.builder()
            .userId(userId)
            .major("unknown")
            .faculty("unknown")
            .degree("unknown")
            .batch("unknown")
            .build();
    }

    @Override
    public List<String> getUserFriends(String userId) {
        log.warn("User Service unavailable. Returning empty friends list for user: {}", userId);
        return Collections.emptyList();
    }
}
