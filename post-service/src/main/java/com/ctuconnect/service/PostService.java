package com.ctuconnect.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import com.ctuconnect.client.MediaServiceClient;
import com.ctuconnect.client.UserServiceClient;
import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.dto.request.PostRequest;
import com.ctuconnect.dto.request.ScheduledPostRequest;
import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.dto.response.PostAnalyticsResponse;
import com.ctuconnect.entity.InteractionEntity;
import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.entity.CommentEntity;
import com.ctuconnect.repository.CommentRepository;
import com.ctuconnect.repository.InteractionRepository;
import com.ctuconnect.repository.PostRepository;
import com.ctuconnect.security.AuthenticatedUser;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

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
    private UserServiceClient userServiceClient;

    @Autowired
    private EventService eventService;

    public PostResponse createPost(PostRequest request, List<MultipartFile> files, String authorId) {
        AuthorInfo author = userServiceClient.getAuthorInfo(authorId);
        if (author == null) {
            throw new RuntimeException("Author not found with id: " + authorId);
        }

        PostEntity post = PostEntity.builder()
                .title(request.getTitle())
                .content(request.getContent())
                .author(author)
                .images(new ArrayList<>())
                .tags(request.getTags() != null ? request.getTags() : new ArrayList<>())
                .category(request.getCategory())
                .privacy(request.getVisibility() != null ? request.getVisibility() : "PUBLIC")
                .stats(new PostEntity.PostStats())
                .build();

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
        Page<PostEntity> posts = postRepository.findAll(pageable);
        
        // Recalculate stats for each post before returning
        posts.forEach(this::recalculatePostStats);
        postRepository.saveAll(posts.getContent());
        
        return posts.map(PostResponse::new);
    }

    public Page<PostResponse> getPostsByAuthor(String authorId, Pageable pageable) {
        System.out.println("DEBUG: PostService.getPostsByAuthor called with authorId: " + authorId);

        Page<PostEntity> posts = postRepository.findByAuthor_Id(authorId, pageable);
        System.out.println("DEBUG: Repository query returned " + posts.getTotalElements() + " posts");

        // Debug first few posts from repository
        posts.getContent().stream().limit(3).forEach(post -> {
            System.out.println("DEBUG: Repository returned post ID: " + post.getId() +
                ", Author ID: " + (post.getAuthor() != null ? post.getAuthor().getId() : "null") +
                ", Author Name: " + (post.getAuthor() != null ? post.getAuthor().getName() : "null"));
        });

        // Recalculate stats for each post before returning
        posts.forEach(this::recalculatePostStats);
        postRepository.saveAll(posts.getContent());
        
        return posts.map(PostResponse::new);
    }

    public Page<PostResponse> getPostsByCategory(String category, Pageable pageable) {
        return postRepository.findByCategory(category, pageable)
                .map(PostResponse::new);
    }

    public Page<PostResponse> searchPosts(String searchTerm, Pageable pageable) {
        return postRepository.findByTitleContainingOrContentContaining(searchTerm, searchTerm, pageable)
                .map(PostResponse::new);
    }

    public PostResponse getPostById(String id, String currentUserId) {
        Optional<PostEntity> postOpt = postRepository.findById(id);
        if (postOpt.isPresent()) {
            PostEntity post = postOpt.get();

            // Record view interaction if user is different from author
            if (currentUserId != null && !currentUserId.equals(post.getAuthorId())) {
                recordViewInteraction(post.getId(), currentUserId);
            }

            // Recalculate stats from database before returning
            recalculatePostStats(post);
            postRepository.save(post);
            
            return new PostResponse(post);
        }
        throw new RuntimeException("Post not found with id: " + id);
    }

    /**
     * Recalculate post stats from actual interactions in database
     * This fixes the issue where stats show 0 even when interactions exist
     */
    private void recalculatePostStats(PostEntity post) {
        // Count actual likes from interactions
        long likeCount = interactionRepository.countByPostIdAndType(post.getId(), InteractionEntity.InteractionType.LIKE);
        long bookmarkCount = interactionRepository.countByPostIdAndType(post.getId(), InteractionEntity.InteractionType.BOOKMARK);
        long shareCount = interactionRepository.countByPostIdAndType(post.getId(), InteractionEntity.InteractionType.SHARE);
        
        // Count comments
        long commentCount = commentRepository.countByPostId(post.getId());
        
        // Update post stats
        post.getStats().setLikes(likeCount);
        post.getStats().setComments(commentCount); 
        post.getStats().setShares(shareCount);
        
        // Update reactions map for LIKE type
        post.getStats().getReactions().put(InteractionEntity.ReactionType.LIKE, (int) likeCount);
    }

    public PostResponse updatePost(String id, PostRequest request, String authorId) {
        Optional<PostEntity> postOpt = postRepository.findById(id);
        if (postOpt.isPresent()) {
            PostEntity post = postOpt.get();

            // Check if user is the author
            if (!post.getAuthorId().equals(authorId)) {
                throw new RuntimeException("Only the author can update this post");
            }

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
            if (request.getVisibility() != null) {
                post.setVisibility(request.getVisibility());
            }

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
        try {
            AuthorInfo author = userServiceClient.getAuthorInfo(userId);
            if (author == null) return;

            // Check if user already viewed this post recently (within last hour)
            Optional<InteractionEntity> existingView = interactionRepository
                    .findByPostIdAndAuthor_IdAndType(postId, userId, InteractionEntity.InteractionType.VIEW);

            if (existingView.isEmpty()) {
                // Create view interaction
                InteractionEntity viewInteraction = new InteractionEntity(postId, author, InteractionEntity.InteractionType.VIEW);
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
        } catch (Exception e) {
            // Log error but don't fail the main operation
            System.err.println("Failed to record view interaction: " + e.getMessage());
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

    /**
     * Enhanced post creation with Facebook-like features
     */
    public PostResponse createEnhancedPost(PostRequest request, AuthenticatedUser user) {
        AuthorInfo author = userServiceClient.getAuthorInfo(user.getId());
        if (author == null) {
            throw new RuntimeException("Author not found with id: " + user.getId());
        }

        PostEntity post = PostEntity.builder()
                .title(request.getTitle())
                .content(request.getContent())
                .author(author)
                .images(request.getImages() != null ? request.getImages() : new ArrayList<>())
                .videos(request.getVideos() != null ? request.getVideos() : new ArrayList<>())
                .tags(request.getTags() != null ? request.getTags() : new ArrayList<>())
                .category(request.getCategory())
                .privacy(request.getVisibility() != null ? request.getVisibility() : "PUBLIC")
                .postType(request.getPostType() != null ? PostEntity.PostType.valueOf(request.getPostType()) : PostEntity.PostType.TEXT)
                .stats(new PostEntity.PostStats())
                .audienceSettings(new PostEntity.AudienceSettings())
                .engagement(new PostEntity.EngagementMetrics())
                .build();

        // Set audience settings if provided
        if (request.getAudienceSettings() != null) {
            post.setAudienceSettings(request.getAudienceSettings());
        }

        // Handle scheduled posts
        if (request.getScheduledAt() != null) {
            post.setScheduledAt(request.getScheduledAt());
            post.setScheduled(true);
        }

        PostEntity savedPost = postRepository.save(post);

        // Publish event
        eventService.publishPostEvent("POST_CREATED", savedPost.getId(), savedPost.getAuthorId(), savedPost);

        return new PostResponse(savedPost);
    }

    /**
     * Get users affected by a post (for cache invalidation)
     */
    public Set<String> getAffectedUserIds(String postId) {
        Optional<PostEntity> postOpt = postRepository.findById(postId);
        if (postOpt.isPresent()) {
            PostEntity post = postOpt.get();
            Set<String> affectedUsers = new HashSet<>();

            // Add author
            affectedUsers.add(post.getAuthorId());

            // Add friends if post is visible to friends
            if ("FRIENDS".equals(post.getPrivacy()) || "PUBLIC".equals(post.getPrivacy())) {
                Set<String> authorFriends = userServiceClient.getFriendIds(post.getAuthorId());
                affectedUsers.addAll(authorFriends);
            }

            return affectedUsers;
        }
        return new HashSet<>();
    }

    /**
     * Handle post interactions (like, comment, share)
     */
    public void handlePostInteraction(String postId, String userId, String action, String reactionType) {
        Optional<PostEntity> postOpt = postRepository.findById(postId);
        if (postOpt.isEmpty()) {
            throw new RuntimeException("Post not found with id: " + postId);
        }

        PostEntity post = postOpt.get();
        AuthorInfo user = userServiceClient.getAuthorInfo(userId);
        if (user == null) {
            throw new RuntimeException("User not found with id: " + userId);
        }

        switch (action.toUpperCase()) {
            case "LIKE":
                handleLikeInteraction(postId, userId, reactionType, post, user);
                break;
            case "UNLIKE":
                handleUnlikeInteraction(postId, userId, post);
                break;
            case "SHARE":
                handleShareInteraction(postId, userId, post, user);
                break;
            default:
                throw new RuntimeException("Unsupported action: " + action);
        }

        // Update engagement metrics
        updateEngagementMetrics(post);
        postRepository.save(post);
    }

    /**
     * Get post author ID
     */
    public String getPostAuthorId(String postId) {
        Optional<PostEntity> postOpt = postRepository.findById(postId);
        if (postOpt.isPresent()) {
            return postOpt.get().getAuthorId();
        }
        throw new RuntimeException("Post not found with id: " + postId);
    }

    /**
     * Add comment to post
     */
    public void addComment(String postId, String userId, String content) {
        Optional<PostEntity> postOpt = postRepository.findById(postId);
        if (postOpt.isEmpty()) {
            throw new RuntimeException("Post not found with id: " + postId);
        }

        AuthorInfo author = userServiceClient.getAuthorInfo(userId);
        if (author == null) {
            throw new RuntimeException("User not found with id: " + userId);
        }

        CommentEntity comment = CommentEntity.builder()
                .postId(postId)
                .author(author)
                .content(content)
                .build();

        commentRepository.save(comment);

        // Update post comment count
        PostEntity post = postOpt.get();
        post.getStats().incrementComments();
        updateEngagementMetrics(post);
        postRepository.save(post);

        // Publish event
        eventService.publishPostEvent("COMMENT_ADDED", postId, userId, comment);
    }

    /**
     * Get post analytics
     */
    public PostAnalyticsResponse getPostAnalytics(String postId, String userId) {
        Optional<PostEntity> postOpt = postRepository.findById(postId);
        if (postOpt.isEmpty()) {
            throw new RuntimeException("Post not found with id: " + postId);
        }

        PostEntity post = postOpt.get();

        // Check if user is the post author
        if (!post.getAuthorId().equals(userId)) {
            throw new RuntimeException("Only post author can view analytics");
        }

        return PostAnalyticsResponse.builder()
                .postId(postId)
                .views(post.getStats().getViews())
                .likes(post.getStats().getLikes())
                .comments(post.getStats().getComments())
                .shares(post.getStats().getShares())
                .engagementRate(post.getEngagement().getEngagementRate())
                .reactions(post.getStats().getReactions())
                .build();
    }

    /**
     * Schedule post for later publishing
     */
    public PostResponse schedulePost(ScheduledPostRequest request, AuthenticatedUser user) {
        AuthorInfo author = userServiceClient.getAuthorInfo(user.getId());
        if (author == null) {
            throw new RuntimeException("Author not found with id: " + user.getId());
        }

        PostEntity post = PostEntity.builder()
                .title(request.getTitle())
                .content(request.getContent())
                .author(author)
                .images(request.getImages() != null ? request.getImages() : new ArrayList<>())
                .tags(request.getTags() != null ? request.getTags() : new ArrayList<>())
                .category(request.getCategory())
                .privacy(request.getVisibility() != null ? request.getVisibility() : "PUBLIC")
                .scheduledAt(request.getScheduledAt())
                .isScheduled(true)
                .stats(new PostEntity.PostStats())
                .build();

        PostEntity savedPost = postRepository.save(post);
        return new PostResponse(savedPost);
    }

    /**
     * Enhanced search with filters
     */
    public List<PostResponse> searchPosts(String query, String category, String faculty,
                                        String dateRange, Pageable pageable) {
        // This would need to be implemented with proper MongoDB queries
        // For now, implementing basic search
        Page<PostEntity> posts;

        if (category != null && !category.isEmpty()) {
            posts = postRepository.findByCategoryAndTitleContainingOrContentContaining(
                category, query, query, pageable);
        } else {
            posts = postRepository.findByTitleContainingOrContentContaining(query, query, pageable);
        }

        return posts.stream()
                .map(PostResponse::new)
                .collect(Collectors.toList());
    }

    /**
     * Update author information in posts (for data consistency)
     */
    public void updateAuthorInfoInPosts(String userId, String fullName, String avatarUrl) {
        List<PostEntity> userPosts = postRepository.findByAuthor_Id(userId);

        for (PostEntity post : userPosts) {
            AuthorInfo updatedAuthor = post.getAuthor();
            updatedAuthor.setFullName(fullName);
            updatedAuthor.setAvatarUrl(avatarUrl);
            post.setAuthor(updatedAuthor);
        }

        if (!userPosts.isEmpty()) {
            postRepository.saveAll(userPosts);
        }
    }

    // Helper methods
    private void handleLikeInteraction(String postId, String userId, String reactionType,
                                     PostEntity post, AuthorInfo user) {
        InteractionEntity.ReactionType reaction =
            reactionType != null ?
            InteractionEntity.ReactionType.valueOf(reactionType.toUpperCase()) :
            InteractionEntity.ReactionType.LIKE;

        // Check if user already reacted
        Optional<InteractionEntity> existingInteraction = interactionRepository
                .findByPostIdAndAuthor_IdAndType(postId, userId, InteractionEntity.InteractionType.REACTION);

        if (existingInteraction.isEmpty()) {
            // Create new reaction
            InteractionEntity interaction = InteractionEntity.builder()
                    .postId(postId)
                    .author(user)
                    .type(InteractionEntity.InteractionType.REACTION)
                    .reactionType(reaction)
                    .build();

            interactionRepository.save(interaction);
            post.getStats().incrementReaction(reaction);
        }
    }

    private void handleUnlikeInteraction(String postId, String userId, PostEntity post) {
        Optional<InteractionEntity> existingInteraction = interactionRepository
                .findByPostIdAndAuthor_IdAndType(postId, userId, InteractionEntity.InteractionType.REACTION);

        if (existingInteraction.isPresent()) {
            InteractionEntity interaction = existingInteraction.get();
            post.getStats().decrementReaction(interaction.getReactionType());
            interactionRepository.delete(interaction);
        }
    }

    private void handleShareInteraction(String postId, String userId, PostEntity post, AuthorInfo user) {
        InteractionEntity interaction = InteractionEntity.builder()
                .postId(postId)
                .author(user)
                .type(InteractionEntity.InteractionType.SHARE)
                .build();

        interactionRepository.save(interaction);
        post.getStats().incrementShares();
    }

    private void updateEngagementMetrics(PostEntity post) {
        PostEntity.EngagementMetrics engagement = post.getEngagement();
        PostEntity.PostStats stats = post.getStats();

        engagement.updateEngagement(
            (int) stats.getLikes(),
            (int) stats.getComments(),
            (int) stats.getShares(),
            (int) stats.getViews()
        );
    }
}
