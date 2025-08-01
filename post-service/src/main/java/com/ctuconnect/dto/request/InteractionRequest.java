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

    private Map<String, Object> metadata = new HashMap<>();
}
