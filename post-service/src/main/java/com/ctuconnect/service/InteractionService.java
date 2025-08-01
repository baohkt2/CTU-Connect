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

    public InteractionResponse createInteraction(String postId, InteractionRequest request, String authorId) {
        AuthorInfo author = userServiceClient.getAuthorInfo(authorId);
        PostEntity post = postRepository.findById(postId)
                .orElseThrow(() -> new RuntimeException("Post not found with id: " + postId));

        InteractionEntity interaction = new InteractionEntity(postId, author, request.getReaction());
        interaction.setMetadata(request.getMetadata());
        InteractionEntity saved = interactionRepository.save(interaction);

        updatePostStats(post, request.getReaction());
        postRepository.save(post);

        eventService.publishInteractionEvent(postId, authorId, request.getReaction().toString());
        return new InteractionResponse(saved);
    }

    private InteractionResponse handleReaction(PostEntity post, AuthorInfo author, InteractionRequest request) {
        InteractionEntity.InteractionType newReaction = request.getReaction();

        Optional<InteractionEntity> existing = interactionRepository.findByPostIdAndAuthorAndType(
                post.getId(), author, InteractionEntity.InteractionType.LIKE);

        if (existing.isPresent()) {
            InteractionEntity current = existing.get();
            InteractionEntity.InteractionType oldReaction = new InteractionEntity().getType();


            if (oldReaction == newReaction) {
                interactionRepository.delete(current);
                post.getStats().decrementReaction(oldReaction.getReactionType());
                postRepository.save(post);

                eventService.publishInteractionEvent(post.getId(), author.getId(), "UN-" + newReaction.name());
                return null;
            } else {
                current.setReaction(newReaction);
                InteractionEntity saved = interactionRepository.save(current);

                post.getStats().decrementReaction(oldReaction.getReactionType());
                post.getStats().incrementReaction(newReaction.getReactionType());
                postRepository.save(post);

                eventService.publishInteractionEvent(post.getId(), author.getId(), newReaction.name());
                return new InteractionResponse(saved);
            }
        } else {
            InteractionEntity newInteraction = new InteractionEntity(post.getId(), author, InteractionEntity.InteractionType.LIKE);
            newInteraction.setReaction(newReaction);
            newInteraction.setMetadata(request.getMetadata());
            InteractionEntity saved = interactionRepository.save(newInteraction);

            post.getStats().incrementReaction(newReaction.getReactionType());
            postRepository.save(post);

            eventService.publishInteractionEvent(post.getId(), author.getId(), newReaction.name());
            return new InteractionResponse(saved);
        }
    }

    private void updatePostStats(PostEntity post, InteractionEntity.InteractionType type) {
        if (Objects.requireNonNull(type) == InteractionEntity.InteractionType.SHARE) {
            post.getStats().incrementShares();
        }
    }

    public boolean hasUserReacted(String postId, String userId) {
        return interactionRepository.findByPostIdAndUserIdAndType(
                postId, userId, InteractionEntity.InteractionType.LIKE).isPresent();
    }

    public long getInteractionCount(String postId, InteractionEntity.InteractionType type) {
        return interactionRepository.countByPostIdAndType(postId, type);
    }
}