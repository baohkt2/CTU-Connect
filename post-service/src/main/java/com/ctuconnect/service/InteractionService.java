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

        // For reaction-based interactions, check if user already has ANY reaction on this post
        if (request.getReaction() == InteractionEntity.InteractionType.LIKE ||
            request.getReaction() == InteractionEntity.InteractionType.REACTION) {

            // Remove any existing reactions from this user on this post
            List<InteractionEntity> existingReactions = interactionRepository
                    .findByPostIdAndAuthor_Id(postId, authorId);

            for (InteractionEntity existing : existingReactions) {
                if (existing.isLike() || existing.isReaction()) {
                    interactionRepository.delete(existing);
                    updatePostStatsOnRemove(post, existing.getType());
                }
            }

            // Check if this is the same reaction being toggled off
            boolean sameReaction = existingReactions.stream()
                    .anyMatch(existing -> isSameReaction(existing, request));

            if (sameReaction) {
                // Same reaction clicked again - just remove it (toggle off)
                postRepository.save(post);
                eventService.publishInteractionEvent(postId, authorId, "UN-" + request.getReaction().toString());
                return new InteractionResponse(false, "Reaction removed");
            }
        } else {
            // For non-reaction interactions (bookmark, share, etc.), check for exact match
            Optional<InteractionEntity> existingInteraction = interactionRepository
                    .findByPostIdAndAuthor_IdAndType(postId, authorId, request.getReaction());

            if (existingInteraction.isPresent()) {
                // Toggle off the exact interaction
                interactionRepository.delete(existingInteraction.get());
                updatePostStatsOnRemove(post, request.getReaction());
                postRepository.save(post);
                eventService.publishInteractionEvent(postId, authorId, "UN-" + request.getReaction().toString());
                return new InteractionResponse(false, "Interaction removed");
            }
        }

        // Create new interaction
        InteractionEntity interaction;
        if (request.getReactionType() != null) {
            interaction = new InteractionEntity(postId, author, request.getReaction(), request.getReactionType());
        } else {
            interaction = new InteractionEntity(postId, author, request.getReaction());
        }

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

    private boolean isSameReaction(InteractionEntity existing, InteractionRequest request) {
        if (existing.getType() == InteractionEntity.InteractionType.LIKE &&
            request.getReaction() == InteractionEntity.InteractionType.LIKE) {
            return true;
        }
        if (existing.getType() == InteractionEntity.InteractionType.REACTION &&
            request.getReaction() == InteractionEntity.InteractionType.REACTION) {
            return Objects.equals(existing.getReactionType(), request.getReactionType());
        }
        return false;
    }

    /**
     * Get user's interaction status for a post
     */
    public InteractionResponse getUserInteractionStatus(String postId, String userId) {
        List<InteractionEntity> userInteractions = interactionRepository.findByPostIdAndAuthor_Id(postId, userId);

        if (userInteractions.isEmpty()) {
            return new InteractionResponse(false, "No interactions found");
        }

        // Return the most recent interaction
        InteractionEntity mostRecent = userInteractions.stream()
                .max((i1, i2) -> i1.getCreatedAt().compareTo(i2.getCreatedAt()))
                .orElse(userInteractions.get(0));

        return new InteractionResponse(mostRecent);
    }

    /**
     * Check if user has liked a specific post
     */
    public boolean hasUserLikedPost(String postId, String userId) {
        List<InteractionEntity> interactions = interactionRepository.findByPostIdAndAuthor_Id(postId, userId);
        return interactions.stream().anyMatch(InteractionEntity::isLike);
    }

    /**
     * Check if user has bookmarked a specific post
     */
    public boolean hasUserBookmarkedPost(String postId, String userId) {
        List<InteractionEntity> interactions = interactionRepository.findByPostIdAndAuthor_Id(postId, userId);
        return interactions.stream().anyMatch(InteractionEntity::isBookmark);
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
            case VIEW:
                post.getStats().incrementViews();
                break;
            case COMMENT:
                post.getStats().incrementComments();
                break;
            case REACTION:
                // Handle reaction stats if needed
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
            case VIEW:
                // Handle view stats if needed
                break;
            case COMMENT:
                post.getStats().decrementComments();
                break;
            case REACTION:
                // Handle reaction stats if needed
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
