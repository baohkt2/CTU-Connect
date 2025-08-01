package com.ctuconnect.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.ctuconnect.entity.InteractionEntity;

import java.util.Map;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class PostAnalyticsResponse {
    private String postId;
    private long views;
    private long likes;
    private long comments;
    private long shares;
    private double engagementRate;
    private Map<InteractionEntity.InteractionType.ReactionType, Integer> reactions;

    // Additional analytics data
    private long totalEngagements;
    private double clickThroughRate;
    private Map<String, Integer> demographicBreakdown;
    private Map<String, Integer> timeBasedEngagement;

    public long getTotalEngagements() {
        return likes + comments + shares;
    }
}
