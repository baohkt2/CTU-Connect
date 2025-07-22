package vn.ctu.edu.postservice.dto.request;

import jakarta.validation.constraints.NotNull;
import vn.ctu.edu.postservice.entity.InteractionEntity;

import java.util.HashMap;
import java.util.Map;

public class CreateInteractionRequest {

    @NotNull(message = "User ID is required")
    private String userId;

    @NotNull(message = "Interaction type is required")
    private InteractionEntity.InteractionType type;

    private Map<String, Object> metadata = new HashMap<>();

    // Getters and Setters
    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public InteractionEntity.InteractionType getType() {
        return type;
    }

    public void setType(InteractionEntity.InteractionType type) {
        this.type = type;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
}
