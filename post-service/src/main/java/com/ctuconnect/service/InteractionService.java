package com.ctuconnect.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ctuconnect.client.UserServiceClient;
import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.dto.request.InteractionRequest;
import com.ctuconnect.dto.response.InteractionResponse;
import com.ctuconnect.entity.InteractionEntity;
import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.repository.InteractionRepository;
import com.ctuconnect.repository.PostRepository;

import java.util.Objects;
import java.util.Optional;
import java.util.List;

@Service
public class InteractionService {

    @Autowired
    private InteractionRepository interactionRepository;

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private UserServiceClient userServiceClient;

    @Autowired
    private EventService eventService;

    /**
     * Create or toggle interaction (like/bookmark)
     * Fixed to properly handle state persistence and prevent duplicates
     */
    public InteractionResponse createInteraction(String postId, InteractionRequest request, String authorId) {
        AuthorInfo author = userServiceClient.getAuthorInfo(authorId);
        if (author == null) {
            throw new RuntimeException("Author not found with id: " + authorId);
        }

        PostEntity post = postRepository.findById(postId)
                .orElseThrow(() -> new RuntimeException("Post not found with id: " + postId));

        // Check if user already has this type of interaction with the post
        Optional<InteractionEntity> existingInteraction = interactionRepository
                .findByPostIdAndAuthor_IdAndType(postId, authorId, request.getReaction());

        if (existingInteraction.isPresent()) {
            // User already has this interaction - remove it (toggle off)
            interactionRepository.delete(existingInteraction.get());

            // Update post stats
            updatePostStatsOnRemove(post, request.getReaction());
            postRepository.save(post);

            eventService.publishInteractionEvent(postId, authorId, "UN-" + request.getReaction().toString());
            return new InteractionResponse(false, "Interaction removed"); // Interaction removed
        } else {
            // Create new interaction
            InteractionEntity interaction = new InteractionEntity(postId, author, request.getReaction());
            if (request.getMetadata() != null) {
                interaction.setMetadata(request.getMetadata());
            }
            InteractionEntity saved = interactionRepository.save(interaction);

            // Update post stats
            updatePostStatsOnAdd(post, request.getReaction());
            postRepository.save(post);

            eventService.publishInteractionEvent(postId, authorId, request.getReaction().toString());
            return new InteractionResponse(saved);
        }
    }

    /**
     * Get user's interaction status for a post
     * This method helps frontend determine current interaction state
     */
    public InteractionResponse getUserInteractionStatus(String postId, String userId) {
        List<InteractionEntity> userInteractions = interactionRepository.findByPostIdAndAuthor_Id(postId, userId);

        if (userInteractions.isEmpty()) {
            return new InteractionResponse(false, "No interactions found");
        }

        // Return the first interaction (in case of multiple, though there shouldn't be)
        return new InteractionResponse(userInteractions.get(0));
    }

    /**
     * Check if user has liked a specific post
     */
    public boolean hasUserLikedPost(String postId, String userId) {
        return interactionRepository.findByPostIdAndAuthor_IdAndType(
            postId, userId, InteractionEntity.InteractionType.LIKE).isPresent();
    }

    /**
     * Check if user has bookmarked a specific post
     */
    public boolean hasUserBookmarkedPost(String postId, String userId) {
        return interactionRepository.findByPostIdAndAuthor_IdAndType(
            postId, userId, InteractionEntity.InteractionType.BOOKMARK).isPresent();
    }

    private void updatePostStatsOnAdd(PostEntity post, InteractionEntity.InteractionType type) {
        switch (type) {
            case LIKE:

                post.getStats().incrementReaction(InteractionEntity.ReactionType.LIKE);
                break;
            case BOOKMARK:
                // Handle bookmark stats if needed
                break;
            case SHARE:
                post.getStats().incrementShares();
                break;
            default:
                break;
        }
    }

    private void updatePostStatsOnRemove(PostEntity post, InteractionEntity.InteractionType type) {
        switch (type) {
            case LIKE:

                post.getStats().decrementReaction(InteractionEntity.ReactionType.LIKE);
                break;
            case BOOKMARK:
                // Handle bookmark stats if needed
                break;
            case SHARE:
                post.getStats().decrementShares();
                break;
            default:
                break;
        }
    }

    public long getInteractionCount(String postId, InteractionEntity.InteractionType type) {
        return interactionRepository.countByPostIdAndType(postId, type);
    }

    /**
     * Check if user has reacted to a post (for legacy compatibility)
     */
    public boolean hasUserReacted(String postId, String userId) {
        return hasUserLikedPost(postId, userId);
    }
}
