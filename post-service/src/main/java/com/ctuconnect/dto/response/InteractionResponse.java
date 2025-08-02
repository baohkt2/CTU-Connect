package com.ctuconnect.dto.response;

import com.ctuconnect.entity.InteractionEntity;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
import java.util.Map;

@Data
public class InteractionResponse {

    // Getters and Setters
    private String id;
    private String postId;
    private String userId;
    private InteractionEntity.InteractionType type;
    private Map<String, Object> metadata;
    private LocalDateTime createdAt;

    // Constructors
    public InteractionResponse() {}

    public InteractionResponse(InteractionEntity interaction) {
        this.id = interaction.getId();
        this.postId = interaction.getPostId();
        this.userId = interaction.getUserId();
        this.type = interaction.getType();
        this.metadata = interaction.getMetadata();
        this.createdAt = interaction.getCreatedAt();
    }

}
