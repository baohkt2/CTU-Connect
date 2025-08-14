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

    public InteractionEntity.ReactionType getReactionType() {
        return reactionType;
    }

    public void setReactionType(InteractionEntity.ReactionType reactionType) {
        this.reactionType = reactionType;
    }
}
