package com.ctuconnect.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;
import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.entity.CommentEntity;
import com.ctuconnect.entity.InteractionEntity;
import com.ctuconnect.repository.PostRepository;
import com.ctuconnect.repository.CommentRepository;
import com.ctuconnect.repository.InteractionRepository;
import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.extern.slf4j.Slf4j;
import java.util.List;
import java.util.Map;

/**
 * Service to handle user profile synchronization across post-service
 * When user updates profile in user-service, this updates author info in posts, comments, interactions
 */
@Service
@Slf4j
public class UserSyncService {

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private CommentRepository commentRepository;

    @Autowired
    private InteractionRepository interactionRepository;

    @Autowired
    private ObjectMapper objectMapper;

    /**
     * Listen for user profile update events from user-service
     */
    @KafkaListener(topics = "user-profile-updated", groupId = "post-service-group")
    public void handleUserProfileUpdate(String message) {
        try {
            log.info("Received user profile update event: {}", message);

            // Parse the message to extract user information
            Map<String, Object> userUpdate = objectMapper.readValue(message, Map.class);
            String userId = (String) userUpdate.get("userId");

            if (userId == null) {
                log.warn("User ID is null in profile update event");
                return;
            }

            // Extract updated user information
            AuthorInfo updatedAuthor = extractAuthorInfo(userUpdate);

            if (updatedAuthor != null) {
                // Update author info in all related entities
                updateAuthorInPosts(userId, updatedAuthor);
                updateAuthorInComments(userId, updatedAuthor);
                updateAuthorInInteractions(userId, updatedAuthor);

                log.info("Successfully updated author info for user: {}", userId);
            } else {
                log.warn("Could not extract author info from user update event");
            }

        } catch (Exception e) {
            log.error("Failed to process user profile update event: {}", e.getMessage(), e);
        }
    }

    /**
     * Update author information in all posts by this user
     */
    private void updateAuthorInPosts(String userId, AuthorInfo updatedAuthor) {
        try {
            List<PostEntity> userPosts = postRepository.findByAuthor_Id(userId);

            for (PostEntity post : userPosts) {
                post.setAuthor(updatedAuthor);
            }

            if (!userPosts.isEmpty()) {
                postRepository.saveAll(userPosts);
                log.info("Updated author info in {} posts for user: {}", userPosts.size(), userId);
            }

        } catch (Exception e) {
            log.error("Failed to update author info in posts for user {}: {}", userId, e.getMessage());
        }
    }

    /**
     * Update author information in all comments by this user
     */
    private void updateAuthorInComments(String userId, AuthorInfo updatedAuthor) {
        try {
            List<CommentEntity> userComments = commentRepository.findByAuthor_Id(userId);

            for (CommentEntity comment : userComments) {
                comment.setAuthor(updatedAuthor);
            }

            if (!userComments.isEmpty()) {
                commentRepository.saveAll(userComments);
                log.info("Updated author info in {} comments for user: {}", userComments.size(), userId);
            }

        } catch (Exception e) {
            log.error("Failed to update author info in comments for user {}: {}", userId, e.getMessage());
        }
    }

    /**
     * Update author information in all interactions by this user
     */
    private void updateAuthorInInteractions(String userId, AuthorInfo updatedAuthor) {
        try {
            List<InteractionEntity> userInteractions = interactionRepository.findByAuthor_Id(userId);

            for (InteractionEntity interaction : userInteractions) {
                interaction.setAuthor(updatedAuthor);
            }

            if (!userInteractions.isEmpty()) {
                interactionRepository.saveAll(userInteractions);
                log.info("Updated author info in {} interactions for user: {}", userInteractions.size(), userId);
            }

        } catch (Exception e) {
            log.error("Failed to update author info in interactions for user {}: {}", userId, e.getMessage());
        }
    }

    /**
     * Extract AuthorInfo from user update event
     */
    private AuthorInfo extractAuthorInfo(Map<String, Object> userUpdate) {
        try {
            String userId = (String) userUpdate.get("userId");
            String fullName = (String) userUpdate.get("fullName");
            String email = (String) userUpdate.get("email");
            String avatarUrl = (String) userUpdate.get("avatarUrl");
            String role = (String) userUpdate.get("role");

            // Handle nested user data structure if it exists
            if (userUpdate.containsKey("userData")) {
                Map<String, Object> userData = (Map<String, Object>) userUpdate.get("userData");
                fullName = fullName != null ? fullName : (String) userData.get("fullName");

                avatarUrl = avatarUrl != null ? avatarUrl : (String) userData.get("avatarUrl");
                role = role != null ? role : (String) userData.get("role");
            }

            if (userId != null && fullName != null) {
                AuthorInfo authorInfo = new AuthorInfo();
                authorInfo.setId(userId);
                authorInfo.setFullName(fullName);
                authorInfo.setAvatarUrl(avatarUrl != null ? avatarUrl : "");
                authorInfo.setRole(role != null ? role : "USER");

                return authorInfo;
            }

        } catch (Exception e) {
            log.error("Failed to extract author info from user update: {}", e.getMessage());
        }

        return null;
    }

    /**
     * Manual sync method for testing or administrative purposes
     */
    public void manualSyncUserProfile(String userId, AuthorInfo updatedAuthor) {
        log.info("Manual sync requested for user: {}", userId);
        updateAuthorInPosts(userId, updatedAuthor);
        updateAuthorInComments(userId, updatedAuthor);
        updateAuthorInInteractions(userId, updatedAuthor);
        log.info("Manual sync completed for user: {}", userId);
    }
}
