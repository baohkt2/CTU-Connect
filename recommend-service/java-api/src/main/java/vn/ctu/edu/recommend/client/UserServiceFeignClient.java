package vn.ctu.edu.recommend.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import vn.ctu.edu.recommend.model.dto.UserDTO;
import vn.ctu.edu.recommend.model.dto.UserAcademicProfile;

/**
 * Feign Client for User Service
 * Communicates with user-service via Eureka service discovery
 */
@FeignClient(name = "user-service", fallback = UserServiceFeignClientFallback.class)
public interface UserServiceFeignClient {

    /**
     * Get user basic information
     */
    @GetMapping("/api/users/{userId}")
    UserDTO getUserById(@PathVariable("userId") String userId);

    /**
     * Get user academic profile
     */
    @GetMapping("/api/users/{userId}/academic-profile")
    UserAcademicProfile getUserAcademicProfile(@PathVariable("userId") String userId);

    /**
     * Get user friends list
     */
    @GetMapping("/api/users/{userId}/friends")
    java.util.List<String> getUserFriends(@PathVariable("userId") String userId);
}
