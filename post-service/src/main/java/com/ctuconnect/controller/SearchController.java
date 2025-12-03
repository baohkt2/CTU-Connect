package com.ctuconnect.controller;

import com.ctuconnect.dto.request.SearchRequest;
import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.dto.response.SearchResponse;
import com.ctuconnect.dto.response.SearchSuggestionResponse;
import com.ctuconnect.service.SearchService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/search")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = "*")
public class SearchController {

    private final SearchService searchService;

    /**
     * Tìm kiếm bài viết với các bộ lọc nâng cao
     */
    @GetMapping("/posts")
    public ResponseEntity<SearchResponse> searchPosts(
            @RequestParam(required = false) String query,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String authorId,
            @RequestParam(required = false) List<String> tags,
            @RequestParam(required = false)
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime dateFrom,
            @RequestParam(required = false)
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime dateTo,
            @RequestParam(defaultValue = "relevance") String sortBy,
            @RequestParam(defaultValue = "desc") String sortOrder,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(required = false) Integer minLikes,
            @RequestParam(required = false) Integer minViews,
            @RequestParam(required = false) Integer minComments,
            @RequestParam(required = false) String postType,
            @RequestParam(required = false) String visibility) {

        try {
            long startTime = System.currentTimeMillis();

            SearchRequest searchRequest = SearchRequest.builder()
                    .query(query)
                    .category(category)
                    .authorId(authorId)
                    .tags(tags)
                    .dateFrom(dateFrom)
                    .dateTo(dateTo)
                    .sortBy(sortBy)
                    .sortOrder(sortOrder)
                    .page(page)
                    .size(size)
                    .minLikes(minLikes)
                    .minViews(minViews)
                    .minComments(minComments)
                    .postType(postType)
                    .visibility(visibility)
                    .build();

            SearchResponse response = searchService.searchPosts(searchRequest);

            long searchTime = System.currentTimeMillis() - startTime;
            response.setSearchTimeMs(searchTime);
            response.setPageSize(size);
            response.setHasNext(page < response.getTotalPages() - 1);
            response.setHasPrevious(page > 0);

            log.info("Search completed in {}ms. Query: '{}', Results: {}",
                    searchTime, query, response.getTotalElements());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error during search: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Tìm kiếm đơn giản chỉ với từ khóa
     */
    @GetMapping("/simple")
    public ResponseEntity<SearchResponse> simpleSearch(
            @RequestParam String q,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        try {
            SearchRequest searchRequest = SearchRequest.builder()
                    .query(q)
                    .page(page)
                    .size(size)
                    .sortBy("relevance")
                    .sortOrder("desc")
                    .build();

            SearchResponse response = searchService.searchPosts(searchRequest);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error during simple search: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Lấy gợi ý tìm kiếm
     */
    @GetMapping("/suggestions")
    public ResponseEntity<SearchSuggestionResponse> getSearchSuggestions(
            @RequestParam String q) {

        try {
            SearchSuggestionResponse suggestions = searchService.getSearchSuggestions(q);
            return ResponseEntity.ok(suggestions);

        } catch (Exception e) {
            log.error("Error getting search suggestions: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Lấy từ khóa trending
     */
    @GetMapping("/trending")
    public ResponseEntity<List<String>> getTrendingTerms() {
        try {
            List<String> trendingTerms = searchService.getTrendingSearchTerms();
            return ResponseEntity.ok(trendingTerms);

        } catch (Exception e) {
            log.error("Error getting trending terms: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Lấy bài viết liên quan
     */
    @GetMapping("/related/{postId}")
    public ResponseEntity<List<PostResponse>> getRelatedPosts(
            @PathVariable String postId,
            @RequestParam(defaultValue = "5") int limit) {

        try {
            List<PostResponse> relatedPosts = searchService.getRelatedPosts(postId, limit);
            return ResponseEntity.ok(relatedPosts);

        } catch (Exception e) {
            log.error("Error getting related posts: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Tìm kiếm bài viết theo danh mục
     */
    @GetMapping("/category/{category}")
    public ResponseEntity<SearchResponse> searchByCategory(
            @PathVariable String category,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "date") String sortBy) {

        try {
            SearchRequest searchRequest = SearchRequest.builder()
                    .category(category)
                    .page(page)
                    .size(size)
                    .sortBy(sortBy)
                    .sortOrder("desc")
                    .build();

            SearchResponse response = searchService.searchPosts(searchRequest);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error searching by category: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Tìm kiếm bài viết theo tag
     */
    @GetMapping("/tag/{tag}")
    public ResponseEntity<SearchResponse> searchByTag(
            @PathVariable String tag,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        try {
            SearchRequest searchRequest = SearchRequest.builder()
                    .tags(List.of(tag))
                    .page(page)
                    .size(size)
                    .sortBy("date")
                    .sortOrder("desc")
                    .build();

            SearchResponse response = searchService.searchPosts(searchRequest);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error searching by tag: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Tìm kiếm bài viết theo tác giả
     */
    @GetMapping("/author/{authorId}")
    public ResponseEntity<SearchResponse> searchByAuthor(
            @PathVariable String authorId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        try {
            SearchRequest searchRequest = SearchRequest.builder()
                    .authorId(authorId)
                    .page(page)
                    .size(size)
                    .sortBy("date")
                    .sortOrder("desc")
                    .build();

            SearchResponse response = searchService.searchPosts(searchRequest);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error searching by author: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }
}
