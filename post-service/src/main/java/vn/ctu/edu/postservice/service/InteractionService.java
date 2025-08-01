package vn.ctu.edu.postservice.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import vn.ctu.edu.postservice.client.UserServiceClient;
import vn.ctu.edu.postservice.dto.AuthorInfo;
import vn.ctu.edu.postservice.dto.request.InteractionRequest;
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

        updatePostStats(post, request.getReaction(), true);
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
            InteractionEntity.InteractionType oldReaction = new InteractionEntity.InteractionType();


            if (oldReaction == newReaction) {
                interactionRepository.delete(current);
                post.getStats().decrementReaction(oldReaction);
                postRepository.save(post);

                eventService.publishInteractionEvent(post.getId(), author.getId(), "UN-" + newReaction.name());
                return null;
            } else {
                current.setReaction(newReaction);
                InteractionEntity saved = interactionRepository.save(current);

                post.getStats().decrementReaction(oldReaction);
                post.getStats().incrementReaction(newReaction);
                postRepository.save(post);

                eventService.publishInteractionEvent(post.getId(), author.getId(), newReaction.name());
                return new InteractionResponse(saved);
            }
        } else {
            InteractionEntity newInteraction = new InteractionEntity(post.getId(), author, InteractionEntity.InteractionType.LIKE);
            newInteraction.setReaction(newReaction);
            newInteraction.setMetadata(request.getMetadata());
            InteractionEntity saved = interactionRepository.save(newInteraction);

            post.getStats().incrementReaction(newReaction);
            postRepository.save(post);

            eventService.publishInteractionEvent(post.getId(), author.getId(), newReaction.name());
            return new InteractionResponse(saved);
        }
    }

    private void updatePostStats(PostEntity post, InteractionEntity.InteractionType type, boolean increment) {
        switch (type) {
            case SHARE:
                if (increment) post.getStats().incrementShares();
                else post.getStats().decrementShares();
                break;
            default:
                break;
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