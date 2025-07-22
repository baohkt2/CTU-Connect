package vn.ctu.edu.postservice.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import vn.ctu.edu.postservice.dto.request.CreateInteractionRequest;
import vn.ctu.edu.postservice.dto.response.InteractionResponse;
import vn.ctu.edu.postservice.entity.InteractionEntity;
import vn.ctu.edu.postservice.entity.PostEntity;
import vn.ctu.edu.postservice.repository.InteractionRepository;
import vn.ctu.edu.postservice.repository.PostRepository;

import java.util.Optional;

@Service
public class InteractionService {

    @Autowired
    private InteractionRepository interactionRepository;

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private EventService eventService;

    public InteractionResponse createInteraction(String postId, CreateInteractionRequest request) {
        // Verify post exists
        Optional<PostEntity> postOpt = postRepository.findById(postId);
        if (postOpt.isEmpty()) {
            throw new RuntimeException("Post not found with id: " + postId);
        }

        // For LIKE interactions, check if already exists and toggle
        if (request.getType() == InteractionEntity.InteractionType.LIKE) {
            return handleLikeInteraction(postId, request.getUserId(), request);
        }

        // For other interactions, create new one
        InteractionEntity interaction = new InteractionEntity(postId, request.getUserId(), request.getType());
        interaction.setMetadata(request.getMetadata());
        InteractionEntity savedInteraction = interactionRepository.save(interaction);

        // Update post stats
        PostEntity post = postOpt.get();
        updatePostStats(post, request.getType(), true);
        postRepository.save(post);

        // Publish event
        eventService.publishInteractionEvent(postId, request.getUserId(), request.getType().toString());

        return new InteractionResponse(savedInteraction);
    }

    private InteractionResponse handleLikeInteraction(String postId, String userId, CreateInteractionRequest request) {
        Optional<InteractionEntity> existingLike = interactionRepository
                .findByPostIdAndUserIdAndType(postId, userId, InteractionEntity.InteractionType.LIKE);

        Optional<PostEntity> postOpt = postRepository.findById(postId);
        PostEntity post = postOpt.get();

        if (existingLike.isPresent()) {
            // Unlike - remove the interaction
            interactionRepository.delete(existingLike.get());
            updatePostStats(post, InteractionEntity.InteractionType.LIKE, false);
            postRepository.save(post);

            eventService.publishInteractionEvent(postId, userId, "UNLIKE");
            return null; // Return null to indicate removal
        } else {
            // Like - create new interaction
            InteractionEntity interaction = new InteractionEntity(postId, userId, InteractionEntity.InteractionType.LIKE);
            interaction.setMetadata(request.getMetadata());
            InteractionEntity savedInteraction = interactionRepository.save(interaction);

            updatePostStats(post, InteractionEntity.InteractionType.LIKE, true);
            postRepository.save(post);

            eventService.publishInteractionEvent(postId, userId, "LIKE");
            return new InteractionResponse(savedInteraction);
        }
    }

    private void updatePostStats(PostEntity post, InteractionEntity.InteractionType type, boolean increment) {
        switch (type) {
            case LIKE:
                if (increment) {
                    post.getStats().incrementLikes();
                } else {
                    post.getStats().decrementLikes();
                }
                break;
            case SHARE:
                if (increment) {
                    post.getStats().incrementShares();
                }
                break;
            // VIEW is handled in PostService
            // BOOKMARK doesn't affect post stats
        }
    }

    public boolean hasUserLikedPost(String postId, String userId) {
        return interactionRepository
                .findByPostIdAndUserIdAndType(postId, userId, InteractionEntity.InteractionType.LIKE)
                .isPresent();
    }

    public boolean hasUserBookmarkedPost(String postId, String userId) {
        return interactionRepository
                .findByPostIdAndUserIdAndType(postId, userId, InteractionEntity.InteractionType.BOOKMARK)
                .isPresent();
    }

    public long getInteractionCount(String postId, InteractionEntity.InteractionType type) {
        return interactionRepository.countByPostIdAndType(postId, type);
    }
}
