package com.ctuconnect.dto.request;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SearchRequest {

    private String query;           // Từ khóa tìm kiếm
    private String category;        // Lọc theo danh mục
    private String authorId;        // Lọc theo tác giả
    private List<String> tags;      // Lọc theo tags
    private LocalDateTime dateFrom; // Tìm kiếm từ ngày
    private LocalDateTime dateTo;   // Tìm kiếm đến ngày
    private String sortBy;          // Sắp xếp theo (relevance, date, popularity)
    private String sortOrder;       // Thứ tự sắp xếp (asc, desc)

    @Builder.Default
    private int page = 0;           // Trang hiện tại

    @Builder.Default
    private int size = 10;          // Số lượng kết quả mỗi trang

    // Các bộ lọc nâng cao
    private Integer minLikes;       // Số like tối thiểu
    private Integer minViews;       // Số view tối thiểu
    private Integer minComments;    // Số comment tối thiểu
    private String postType;        // Loại bài viết
    private String visibility;      // Độ hiển thị
    private List<String> faculties; // Lọc theo khoa
    private List<String> majors;    // Lọc theo chuyên ngành
}
