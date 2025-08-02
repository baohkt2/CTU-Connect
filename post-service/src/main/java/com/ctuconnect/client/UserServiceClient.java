package com.ctuconnect.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import com.ctuconnect.dto.AuthorInfo;

import java.util.Set;

@FeignClient(name = "user-service", url = "${user-service.url:http://localhost:8080}")
public interface UserServiceClient {
    @GetMapping("/api/users/sync/authors/{id}")
    AuthorInfo getAuthorInfo(@PathVariable("id") String authorId);

    @GetMapping("/api/users/{userId}/friends/ids")
    Set<String> getFriendIds(@PathVariable("userId") String userId);

    // Additional methods needed by NewsFeedService
    @GetMapping("/api/users/{userId}/close-interactions")
    Set<String> getCloseInteractionIds(@PathVariable("userId") String userId);

    @GetMapping("/api/users/{userId}/same-faculty")
    Set<String> getSameFacultyUserIds(@PathVariable("userId") String userId);

    @GetMapping("/api/users/{userId}/same-major")
    Set<String> getSameMajorUserIds(@PathVariable("userId") String userId);

    @GetMapping("/api/users/{userId}/interest-tags")
    Set<String> getUserInterestTags(@PathVariable("userId") String userId);

    @GetMapping("/api/users/{userId}/preferred-categories")
    Set<String> getUserPreferredCategories(@PathVariable("userId") String userId);

    @GetMapping("/api/users/{userId}/faculty-id")
    String getUserFacultyId(@PathVariable("userId") String userId);

    @GetMapping("/api/users/{userId}/major-id")
    String getUserMajorId(@PathVariable("userId") String userId);
}
