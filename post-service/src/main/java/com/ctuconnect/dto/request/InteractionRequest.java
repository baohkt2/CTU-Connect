package com.ctuconnect.dto.request;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.ctuconnect.entity.InteractionEntity;

import java.util.HashMap;
import java.util.Map;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class InteractionRequest {

    @NotNull(message = "Interaction type is required")
    private InteractionEntity.InteractionType reaction;

    private InteractionEntity.ReactionType reactionType;

    private Map<String, Object> metadata = new HashMap<>();

    // Getter and setter methods for compatibility
    public InteractionEntity.InteractionType getReaction() {
        return reaction;
    }

    public void setReaction(InteractionEntity.InteractionType reaction) {
        this.reaction = reaction;
    }

    public InteractionEntity.ReactionType getReactionType() {
        return reactionType;
    }

    public void setReactionType(InteractionEntity.ReactionType reactionType) {
        this.reactionType = reactionType;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    public void setType(String s) {
        if (s != null) {
            try {
                this.reaction = InteractionEntity.InteractionType.valueOf(s.toUpperCase());
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("Invalid interaction type: " + s);
            }
        } else {
            throw new IllegalArgumentException("Interaction type cannot be null");
        }
    }
}
