package com.ctuconnect.service;

import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.dto.FriendSuggestionDTO;
import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.neo4j.core.Neo4jTemplate;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
@Slf4j
public class SocialGraphService {

    private final UserRepository userRepository;
    private final Neo4jTemplate neo4jTemplate;
    private final RedisTemplate<String, Object> redisTemplate;

    private static final String FRIEND_SUGGESTIONS_CACHE = "friend_suggestions:";
    private static final String MUTUAL_FRIENDS_CACHE = "mutual_friends:";
    private static final int CACHE_TTL_HOURS = 6;

    /**
     * Facebook-like friend suggestion algorithm using multiple signals
     */
    public List<FriendSuggestionDTO> getFriendSuggestions(String userId, int limit) {
        String cacheKey = FRIEND_SUGGESTIONS_CACHE + userId;

        // Try cache first
        List<FriendSuggestionDTO> cached = getCachedSuggestions(cacheKey);
        if (cached != null && !cached.isEmpty()) {
            return cached.stream().limit(limit).collect(Collectors.toList());
        }

        UserEntity currentUser = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        List<FriendSuggestionDTO> suggestions = new ArrayList<>();

        // 1. Mutual friends suggestions (highest priority)
        suggestions.addAll(getMutualFriendsSuggestions(currentUser, 20));

        // 2. Academic connections (same faculty, major, batch)
        suggestions.addAll(getAcademicConnectionSuggestions(currentUser, 15));

        // 3. Friends of friends
        suggestions.addAll(getFriendsOfFriendsSuggestions(currentUser, 15));

        // 4. People who viewed your profile
        suggestions.addAll(getProfileViewersSuggestions(currentUser, 10));

        // 5. Similar interests/activities
        suggestions.addAll(getSimilarInterestsSuggestions(currentUser, 10));

        // Remove duplicates and rank by relevance score
        Map<String, FriendSuggestionDTO> uniqueSuggestions = suggestions.stream()
                .filter(suggestion -> !suggestion.getUserId().equals(userId))
                .filter(suggestion -> !isAlreadyConnected(userId, suggestion.getUserId()))
                .collect(Collectors.toMap(
                    FriendSuggestionDTO::getUserId,
                    suggestion -> suggestion,
                    (existing, replacement) -> {
                        // Keep the one with higher relevance score
                        return existing.getRelevanceScore() > replacement.getRelevanceScore()
                            ? existing : replacement;
                    }
                ));

        List<FriendSuggestionDTO> rankedSuggestions = uniqueSuggestions.values().stream()
                .sorted((a, b) -> Double.compare(b.getRelevanceScore(), a.getRelevanceScore()))
                .limit(limit)
                .collect(Collectors.toList());

        // Cache the results
        cacheSuggestions(cacheKey, rankedSuggestions);

        return rankedSuggestions;
    }

    /**
     * Get mutual friends suggestions with high relevance
     */
    private List<FriendSuggestionDTO> getMutualFriendsSuggestions(UserEntity user, int limit) {
        // Use repository method instead of direct Neo4j queries
        try {
            List<UserEntity> mutualFriendSuggestions = userRepository.findFriendSuggestions(user.getId());

            return mutualFriendSuggestions.stream()
                    .limit(limit)
                    .map(suggestedUser -> {
                        int mutualCount = getMutualFriendsCount(user.getId(), suggestedUser.getId());

                        return FriendSuggestionDTO.builder()
                                .userId(suggestedUser.getId())
                                .username(suggestedUser.getUsername())
                                .fullName(suggestedUser.getFullName())
                                .avatarUrl(suggestedUser.getAvatarUrl())
                                .mutualFriendsCount(mutualCount)
                                .suggestionReason("You have " + mutualCount + " mutual friends")
                                .relevanceScore(calculateMutualFriendsScore(mutualCount))
                                .suggestionType(FriendSuggestionDTO.SuggestionType.MUTUAL_FRIENDS)
                                .build();
                    })
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.warn("Failed to get mutual friends suggestions: {}", e.getMessage());
            return new ArrayList<>();
        }
    }

    /**
     * Academic connections (same faculty, major, batch)
     */
    private List<FriendSuggestionDTO> getAcademicConnectionSuggestions(UserEntity user, int limit) {
        List<FriendSuggestionDTO> suggestions = new ArrayList<>();

        try {
            // Get users from same faculty
            if (user.getFacultyId() != null) {
                List<UserEntity> sameFacultyUsers = userRepository.findUsersByFacultyId(user.getFacultyId());
                suggestions.addAll(sameFacultyUsers.stream()
                        .filter(u -> !u.getId().equals(user.getId()))
                        .limit(limit / 3)
                        .map(suggestedUser -> FriendSuggestionDTO.builder()
                                .userId(suggestedUser.getId())
                                .username(suggestedUser.getUsername())
                                .fullName(suggestedUser.getFullName())
                                .avatarUrl(suggestedUser.getAvatarUrl())
                                .suggestionReason("Same faculty: " + user.getFacultyName())
                                .relevanceScore(calculateAcademicScore(2))
                                .suggestionType(FriendSuggestionDTO.SuggestionType.ACADEMIC_CONNECTION)
                                .build())
                        .collect(Collectors.toList()));
            }

            // Get users from same major
            if (user.getMajorId() != null) {
                List<UserEntity> sameMajorUsers = userRepository.findUsersByMajorId(user.getMajorId());
                suggestions.addAll(sameMajorUsers.stream()
                        .filter(u -> !u.getId().equals(user.getId()))
                        .limit(limit / 3)
                        .map(suggestedUser -> FriendSuggestionDTO.builder()
                                .userId(suggestedUser.getId())
                                .username(suggestedUser.getUsername())
                                .fullName(suggestedUser.getFullName())
                                .avatarUrl(suggestedUser.getAvatarUrl())
                                .suggestionReason("Same major: " + user.getMajorName())
                                .relevanceScore(calculateAcademicScore(3))
                                .suggestionType(FriendSuggestionDTO.SuggestionType.ACADEMIC_CONNECTION)
                                .build())
                        .collect(Collectors.toList()));
            }
        } catch (Exception e) {
            log.warn("Failed to get academic connection suggestions: {}", e.getMessage());
        }

        return suggestions;
    }

    /**
     * Friends of friends suggestions
     */
    private List<FriendSuggestionDTO> getFriendsOfFriendsSuggestions(UserEntity user, int limit) {
        try {
            // Get friends of the current user
            List<UserEntity> friends = userRepository.findFriends(user.getId());
            List<FriendSuggestionDTO> suggestions = new ArrayList<>();

            for (UserEntity friend : friends) {
                List<UserEntity> friendsOfFriend = userRepository.findFriends(friend.getId());

                for (UserEntity suggestion : friendsOfFriend) {
                    if (!suggestion.getId().equals(user.getId()) &&
                        !userRepository.areFriends(user.getId(), suggestion.getId())) {

                        suggestions.add(FriendSuggestionDTO.builder()
                                .userId(suggestion.getId())
                                .username(suggestion.getUsername())
                                .fullName(suggestion.getFullName())
                                .avatarUrl(suggestion.getAvatarUrl())
                                .suggestionReason("Friend of " + friend.getFullName())
                                .relevanceScore(calculateFriendsOfFriendsScore(1))
                                .suggestionType(FriendSuggestionDTO.SuggestionType.FRIENDS_OF_FRIENDS)
                                .build());
                    }
                }
            }

            return suggestions.stream()
                    .limit(limit)
                    .collect(Collectors.toList());

        } catch (Exception e) {
            log.warn("Failed to get friends of friends suggestions: {}", e.getMessage());
            return new ArrayList<>();
        }
    }

    /**
     * People who viewed your profile
     */
    private List<FriendSuggestionDTO> getProfileViewersSuggestions(UserEntity user, int limit) {
        // This would typically require profile view tracking
        // For now, returning empty list as this feature requires additional implementation
        return new ArrayList<>();
    }

    /**
     * Similar interests suggestions
     */
    private List<FriendSuggestionDTO> getSimilarInterestsSuggestions(UserEntity user, int limit) {
        // Implementation would analyze user interactions, tags, groups, etc.
        // For now, returning empty list as it requires more complex analysis
        return new ArrayList<>();
    }

    /**
     * Calculate mutual friends count between two users
     */
    public int getMutualFriendsCount(String userId1, String userId2) {
        String cacheKey = MUTUAL_FRIENDS_CACHE + userId1 + ":" + userId2;

        Object cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            return ((Number) cached).intValue();
        }

        try {
            List<UserEntity> mutualFriends = userRepository.findMutualFriends(userId1, userId2);
            int count = mutualFriends.size();

            // Cache for 1 hour
            redisTemplate.opsForValue().set(cacheKey, count, 1, TimeUnit.HOURS);

            return count;
        } catch (Exception e) {
            log.warn("Failed to get mutual friends count: {}", e.getMessage());
            return 0;
        }
    }

    private boolean isAlreadyConnected(String userId1, String userId2) {
        try {
            return userRepository.areFriends(userId1, userId2) ||
                   userRepository.hasPendingFriendRequest(userId1, userId2) ||
                   userRepository.hasPendingFriendRequest(userId2, userId1);
        } catch (Exception e) {
            log.warn("Failed to check connection status: {}", e.getMessage());
            return false;
        }
    }

    private double calculateMutualFriendsScore(int mutualCount) {
        // Higher mutual friends = higher relevance
        return Math.min(1.0, mutualCount / 10.0);
    }

    private double calculateAcademicScore(int academicScore) {
        return academicScore / 6.0; // Max score is 6 (3+2+1)
    }

    private double calculateFriendsOfFriendsScore(int paths) {
        return Math.min(0.7, paths / 5.0);
    }

    private String buildAcademicReason(UserEntity user, UserEntity suggestion) {
        List<String> connections = new ArrayList<>();

        if (Objects.equals(user.getBatchId(), suggestion.getBatchId())) {
            connections.add("same batch");
        }
        if (Objects.equals(user.getMajorId(), suggestion.getMajorId())) {
            connections.add("same major");
        }
        if (Objects.equals(user.getFacultyId(), suggestion.getFacultyId())) {
            connections.add("same faculty");
        }

        return "You share " + String.join(", ", connections);
    }

    private List<FriendSuggestionDTO> getCachedSuggestions(String cacheKey) {
        try {
            return (List<FriendSuggestionDTO>) redisTemplate.opsForValue().get(cacheKey);
        } catch (Exception e) {
            log.warn("Failed to get cached suggestions: {}", e.getMessage());
            return null;
        }
    }

    private void cacheSuggestions(String cacheKey, List<FriendSuggestionDTO> suggestions) {
        try {
            redisTemplate.opsForValue().set(cacheKey, suggestions, CACHE_TTL_HOURS, TimeUnit.HOURS);
        } catch (Exception e) {
            log.warn("Failed to cache suggestions: {}", e.getMessage());
        }
    }

    /**
     * Invalidate friend suggestions cache when user relationships change
     */
    public void invalidateFriendSuggestionsCache(String userId) {
        String pattern = FRIEND_SUGGESTIONS_CACHE + userId;
        redisTemplate.delete(pattern);

        // Also invalidate mutual friends cache for this user
        String mutualPattern = MUTUAL_FRIENDS_CACHE + userId + ":*";
        Set<String> keys = redisTemplate.keys(mutualPattern);
        if (keys != null && !keys.isEmpty()) {
            redisTemplate.delete(keys);
        }
    }
}
