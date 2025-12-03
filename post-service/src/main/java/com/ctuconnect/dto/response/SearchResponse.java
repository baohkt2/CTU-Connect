package com.ctuconnect.dto.response;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SearchResponse {

    private List<PostResponse> posts;           // Danh sách bài viết tìm được
    private long totalElements;                 // Tổng số kết quả
    private int totalPages;                     // Tổng số trang
    private int currentPage;                    // Trang hiện tại
    private int pageSize;                       // Kích thước trang
    private String searchQuery;                 // Từ khóa tìm kiếm
    private Map<String, Object> filtersApplied; // Các bộ lọc đã áp dụng
    private long searchTimeMs;                  // Thời gian tìm kiếm (milliseconds)

    // Thông tin bổ sung
    private List<String> suggestions;           // Gợi ý tìm kiếm
    private Map<String, Long> facetCounts;      // Số lượng theo từng facet
    private boolean hasNext;                    // Có trang tiếp theo không
    private boolean hasPrevious;               // Có trang trước không
}
