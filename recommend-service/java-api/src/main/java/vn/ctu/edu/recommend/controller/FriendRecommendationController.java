package vn.ctu.edu.recommend.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import vn.ctu.edu.recommend.model.dto.FriendRecommendationResponse;
import vn.ctu.edu.recommend.service.HybridFriendRecommendationService;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import java.util.Map;

/**
 * REST Controller for Friend Recommendation API
 * 
 * Provides ML-enhanced friend suggestions using hybrid ranking
 */
@RestController
@RequestMapping("/api/recommendations/friends")
@Tag(name = "Friend Recommendations", description = "ML-enhanced friend suggestion API")
@RequiredArgsConstructor
@Slf4j
@Validated
public class FriendRecommendationController {

    private final HybridFriendRecommendationService friendRecommendationService;

    /**
     * Get personalized friend suggestions for a user
     */
    @GetMapping("/{userId}")
    @Operation(
        summary = "Get friend suggestions",
        description = "Returns ML-ranked friend suggestions based on content similarity, mutual friends, academic connections, and activity"
    )
    @ApiResponses({
        @ApiResponse(
            responseCode = "200",
            description = "Suggestions retrieved successfully",
            content = @Content(schema = @Schema(implementation = FriendRecommendationResponse.class))
        ),
        @ApiResponse(responseCode = "400", description = "Invalid parameters"),
        @ApiResponse(responseCode = "500", description = "Internal server error")
    })
    public ResponseEntity<FriendRecommendationResponse> getFriendSuggestions(
            @PathVariable
            @NotBlank(message = "User ID is required")
            @Parameter(description = "ID of the user to get suggestions for")
            String userId,
            
            @RequestParam(defaultValue = "20")
            @Min(value = 1, message = "Limit must be at least 1")
            @Max(value = 100, message = "Limit cannot exceed 100")
            @Parameter(description = "Maximum number of suggestions to return")
            int limit
    ) {
        log.info("📥 Friend suggestion request - userId: {}, limit: {}", userId, limit);
        
        FriendRecommendationResponse response = friendRecommendationService.getFriendSuggestions(userId, limit);
        
        log.info("📤 Returning {} friend suggestions for user: {} (source: {})", 
            response.getCount(), userId, response.getMetadata().getSource());
        
        return ResponseEntity.ok(response);
    }

    /**
     * Record user feedback on a suggestion
     */
    @PostMapping("/{userId}/feedback")
    @Operation(
        summary = "Record feedback",
        description = "Record user interaction with a friend suggestion (click, request, accept, reject, dismiss)"
    )
    @ApiResponses({
        @ApiResponse(responseCode = "200", description = "Feedback recorded successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid parameters")
    })
    public ResponseEntity<Map<String, Object>> recordFeedback(
            @PathVariable
            @NotBlank(message = "User ID is required")
            String userId,
            
            @RequestBody
            FeedbackRequest request
    ) {
        log.info("📝 Recording feedback - userId: {}, recommendedUserId: {}, action: {}", 
            userId, request.recommendedUserId(), request.action());
        
        friendRecommendationService.recordFeedback(userId, request.recommendedUserId(), request.action());
        
        return ResponseEntity.ok(Map.of(
            "success", true,
            "message", "Feedback recorded successfully"
        ));
    }

    /**
     * Invalidate cache for a user
     */
    @DeleteMapping("/{userId}/cache")
    @Operation(
        summary = "Invalidate cache",
        description = "Clear cached friend suggestions for a user"
    )
    public ResponseEntity<Map<String, Object>> invalidateCache(
            @PathVariable
            @NotBlank(message = "User ID is required")
            String userId
    ) {
        log.info("🗑️ Invalidating cache for user: {}", userId);
        
        friendRecommendationService.invalidateCache(userId);
        
        return ResponseEntity.ok(Map.of(
            "success", true,
            "message", "Cache invalidated successfully"
        ));
    }

    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    @Operation(summary = "Health check", description = "Check if the friend recommendation service is running")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        return ResponseEntity.ok(Map.of(
            "status", "UP",
            "service", "friend-recommendation",
            "timestamp", System.currentTimeMillis()
        ));
    }

    /**
     * Record DTO for feedback requests
     */
    public record FeedbackRequest(
        @NotBlank(message = "Recommended user ID is required")
        String recommendedUserId,
        
        @NotBlank(message = "Action is required")
        String action
    ) {}
}
