package com.ctuconnect.dto.response;

import com.ctuconnect.entity.InteractionEntity;
import lombok.*;

import java.time.LocalDateTime;
import java.util.Map;

@Data
@Builder
@AllArgsConstructor
public class InteractionResponse {

    // Getters and Setters
    private String id;
    private String postId;
    private String userId;
    private InteractionEntity.InteractionType type;
    private Map<String, Object> metadata;
    private LocalDateTime createdAt;

    // New fields for status responses
    private boolean hasInteraction;
    private String message;

    // Constructors
    public InteractionResponse() {}

    public InteractionResponse(InteractionEntity interaction) {
        this.id = interaction.getId();
        this.postId = interaction.getPostId();
        this.userId = interaction.getUserId();
        this.type = interaction.getType();
        this.metadata = interaction.getMetadata();
        this.createdAt = interaction.getCreatedAt();
        this.hasInteraction = true;
    }

    // Constructor for status responses
    public InteractionResponse(boolean hasInteraction, String message) {
        this.hasInteraction = hasInteraction;
        this.message = message;
    }

}
