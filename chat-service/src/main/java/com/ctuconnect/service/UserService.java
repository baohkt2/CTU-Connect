package com.ctuconnect.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Service
@Slf4j
public class UserService {

    private final RestTemplate restTemplate;
    private final String userServiceUrl;

    public UserService(RestTemplate restTemplate,
                      @Value("${user.service.url:http://user-service:8081}") String userServiceUrl) {
        this.restTemplate = restTemplate;
        this.userServiceUrl = userServiceUrl;
    }

    public Map<String, Object> getUserInfo(String userId) {
        try {
            String url = userServiceUrl + "/api/users/" + userId;
            return restTemplate.getForObject(url, Map.class);
        } catch (Exception e) {
            log.error("Failed to get user info for userId: {}", userId, e);
            return createDefaultUserInfo(userId);
        }
    }

    public Map<String, Object> getUsersByIds(java.util.List<String> userIds) {
        try {
            String url = userServiceUrl + "/api/users/batch";
            Map<String, Object> request = Map.of("userIds", userIds);
            return restTemplate.postForObject(url, request, Map.class);
        } catch (Exception e) {
            log.error("Failed to get users info for userIds: {}", userIds, e);
            return Map.of("users", java.util.Collections.emptyList());
        }
    }

    private Map<String, Object> createDefaultUserInfo(String userId) {
        return Map.of(
            "id", userId,
            "name", "User " + userId,
            "avatar", "",
            "fullName", "User " + userId
        );
    }
}
