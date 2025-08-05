package com.ctuconnect.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UpdateConversationRequest {
    private String name;
    private String description;
    private String avatarUrl;
    private List<String> participantIds;
}
