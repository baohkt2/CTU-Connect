package vn.ctu.edu.postservice.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import vn.ctu.edu.postservice.client.MediaServiceClient;
import vn.ctu.edu.postservice.dto.request.CreatePostRequest;
import vn.ctu.edu.postservice.dto.request.UpdatePostRequest;
import vn.ctu.edu.postservice.dto.response.PostResponse;
import vn.ctu.edu.postservice.entity.InteractionEntity;
import vn.ctu.edu.postservice.entity.PostEntity;
import vn.ctu.edu.postservice.repository.CommentRepository;
import vn.ctu.edu.postservice.repository.InteractionRepository;
import vn.ctu.edu.postservice.repository.PostRepository;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
public class PostService {

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private CommentRepository commentRepository;

    @Autowired
    private InteractionRepository interactionRepository;

    @Autowired
    private MediaServiceClient mediaServiceClient;

    @Autowired
    private EventService eventService;

    public PostResponse createPost(CreatePostRequest request, List<MultipartFile> files) {
        PostEntity post = new PostEntity(request.getTitle(), request.getContent(), request.getAuthorId());
        post.setTags(request.getTags());
        post.setCategory(request.getCategory());

        // Upload files to media-service
        if (files != null && !files.isEmpty()) {
            List<String> imageUrls = new ArrayList<>();
            for (MultipartFile file : files) {
                if (!file.isEmpty()) {
                    try {
                        MediaServiceClient.MediaUploadResponse uploadResponse =
                            mediaServiceClient.uploadFile(file, getFileType(file));
                        imageUrls.add(uploadResponse.getFileUrl());
                    } catch (Exception e) {
                        // Log error but continue processing
                        System.err.println("Failed to upload file: " + e.getMessage());
                    }
                }
            }
            post.setImages(imageUrls);
        }

        PostEntity savedPost = postRepository.save(post);

        // Publish event
        eventService.publishPostEvent("POST_CREATED", savedPost.getId(), savedPost.getAuthorId(), savedPost);

        return new PostResponse(savedPost);
    }

    public Page<PostResponse> getAllPosts(Pageable pageable) {
        return postRepository.findAll(pageable)
                .map(PostResponse::new);
    }

    public Page<PostResponse> getPostsByAuthor(String authorId, Pageable pageable) {
        return postRepository.findByAuthorId(authorId, pageable)
                .map(PostResponse::new);
    }

    public Page<PostResponse> getPostsByCategory(String category, Pageable pageable) {
        return postRepository.findByCategory(category, pageable)
                .map(PostResponse::new);
    }

    public Page<PostResponse> searchPosts(String searchTerm, Pageable pageable) {
        return postRepository.findByTitleOrContentContaining(searchTerm, pageable)
                .map(PostResponse::new);
    }

    public PostResponse getPostById(String id, String userId) {
        Optional<PostEntity> postOpt = postRepository.findById(id);
        if (postOpt.isPresent()) {
            PostEntity post = postOpt.get();

            // Record view interaction if userId is provided
            if (userId != null && !userId.equals(post.getAuthorId())) {
                recordViewInteraction(post.getId(), userId);
            }

            return new PostResponse(post);
        }
        throw new RuntimeException("Post not found with id: " + id);
    }

    public PostResponse updatePost(String id, UpdatePostRequest request, String authorId) {
        Optional<PostEntity> postOpt = postRepository.findById(id);
        if (postOpt.isPresent()) {
            PostEntity post = postOpt.get();

            // Check if user is the author
            if (!post.getAuthorId().equals(authorId)) {
                throw new RuntimeException("Only the author can update this post");
            }

            // Update fields
            if (request.getTitle() != null) {
                post.setTitle(request.getTitle());
            }
            if (request.getContent() != null) {
                post.setContent(request.getContent());
            }
            if (request.getTags() != null) {
                post.setTags(request.getTags());
            }
            if (request.getCategory() != null) {
                post.setCategory(request.getCategory());
            }

            post.updateTimestamp();
            PostEntity savedPost = postRepository.save(post);

            // Publish event
            eventService.publishPostEvent("POST_UPDATED", savedPost.getId(), savedPost.getAuthorId(), savedPost);

            return new PostResponse(savedPost);
        }
        throw new RuntimeException("Post not found with id: " + id);
    }

    public void deletePost(String id, String authorId) {
        Optional<PostEntity> postOpt = postRepository.findById(id);
        if (postOpt.isPresent()) {
            PostEntity post = postOpt.get();

            // Check if user is the author
            if (!post.getAuthorId().equals(authorId)) {
                throw new RuntimeException("Only the author can delete this post");
            }

            // Delete associated comments and interactions
            commentRepository.deleteByPostId(id);
            interactionRepository.deleteByPostId(id);

            // Delete the post
            postRepository.deleteById(id);

            // Publish event
            eventService.publishPostEvent("POST_DELETED", id, authorId, post);
        } else {
            throw new RuntimeException("Post not found with id: " + id);
        }
    }

    private void recordViewInteraction(String postId, String userId) {
        // Check if user already viewed this post recently
        Optional<InteractionEntity> existingView = interactionRepository
                .findByPostIdAndUserIdAndType(postId, userId, InteractionEntity.InteractionType.VIEW);

        if (existingView.isEmpty()) {
            // Create view interaction
            InteractionEntity viewInteraction = new InteractionEntity(postId, userId, InteractionEntity.InteractionType.VIEW);
            interactionRepository.save(viewInteraction);

            // Update post stats
            Optional<PostEntity> postOpt = postRepository.findById(postId);
            if (postOpt.isPresent()) {
                PostEntity post = postOpt.get();
                post.getStats().incrementViews();
                postRepository.save(post);
            }

            // Publish interaction event
            eventService.publishInteractionEvent(postId, userId, "VIEW");
        }
    }

    private String getFileType(MultipartFile file) {
        String contentType = file.getContentType();
        if (contentType != null) {
            if (contentType.startsWith("image/")) return "IMAGE";
            if (contentType.startsWith("video/")) return "VIDEO";
            if (contentType.startsWith("audio/")) return "AUDIO";
            if (contentType.equals("application/pdf")) return "PDF";
        }
        return "DOCUMENT";
    }

    public List<PostResponse> getTopViewedPosts() {
        return postRepository.findTop10ByOrderByStatsViewsDesc()
                .stream()
                .map(PostResponse::new)
                .toList();
    }

    public List<PostResponse> getTopLikedPosts() {
        return postRepository.findTop10ByOrderByStatsLikesDesc()
                .stream()
                .map(PostResponse::new)
                .toList();
    }
}
