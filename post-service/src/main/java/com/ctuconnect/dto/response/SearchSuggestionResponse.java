package com.ctuconnect.dto.response;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.util.List;
import java.util.ArrayList;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SearchSuggestionResponse {

    @Builder.Default
    private List<String> titleSuggestions = new ArrayList<>();      // Gợi ý từ tiêu đề

    @Builder.Default
    private List<String> categorySuggestions = new ArrayList<>();   // Gợi ý danh mục

    @Builder.Default
    private List<String> tagSuggestions = new ArrayList<>();        // Gợi ý tags

    @Builder.Default
    private List<String> authorSuggestions = new ArrayList<>();     // Gợi ý tác giả

    @Builder.Default
    private List<String> trendingSuggestions = new ArrayList<>();   // Từ khóa trending
}
