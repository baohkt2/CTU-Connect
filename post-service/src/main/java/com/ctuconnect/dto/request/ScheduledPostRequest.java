package com.ctuconnect.dto.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ScheduledPostRequest {
    private String title;
    private String content;
    private List<String> images;
    private List<String> videos;
    private List<String> tags;
    private String category;
    private String visibility;
    private LocalDateTime scheduledAt;
}
