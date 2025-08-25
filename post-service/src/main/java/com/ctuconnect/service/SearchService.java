package com.ctuconnect.service;

import com.ctuconnect.dto.request.SearchRequest;
import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.dto.response.SearchResponse;
import com.ctuconnect.dto.response.SearchSuggestionResponse;
import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.repository.PostRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.aggregation.Aggregation;
import org.springframework.data.mongodb.core.aggregation.AggregationResults;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class SearchService {

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private MongoTemplate mongoTemplate;

    /**
     * Advanced search with multiple filters and relevance scoring
     */
    public SearchResponse searchPosts(SearchRequest searchRequest) {
        Query query = buildSearchQuery(searchRequest);

        // Apply sorting based on relevance and other criteria
        Sort sort = buildSort(searchRequest);
        query.with(sort);

        // Apply pagination
        Pageable pageable = PageRequest.of(
            searchRequest.getPage(),
            searchRequest.getSize()
        );
        query.with(pageable);

        List<PostEntity> posts = mongoTemplate.find(query, PostEntity.class);
        long totalCount = mongoTemplate.count(Query.of(query).limit(-1).skip(-1), PostEntity.class);

        // Convert to response DTOs
        List<PostResponse> postResponses = posts.stream()
            .map(PostResponse::new)
            .collect(Collectors.toList());

        // Build search response with metadata
        SearchResponse response = new SearchResponse();
        response.setPosts(postResponses);
        response.setTotalElements(totalCount);
        response.setTotalPages((int) Math.ceil((double) totalCount / searchRequest.getSize()));
        response.setCurrentPage(searchRequest.getPage());
        response.setSearchQuery(searchRequest.getQuery());
        response.setFiltersApplied(buildFiltersApplied(searchRequest));

        return response;
    }

    /**
     * Get search suggestions based on partial input
     */
    public SearchSuggestionResponse getSearchSuggestions(String partialQuery) {
        SearchSuggestionResponse response = new SearchSuggestionResponse();

        if (partialQuery == null || partialQuery.trim().length() < 2) {
            return response;
        }

        String query = partialQuery.trim();

        // Get title suggestions
        List<String> titleSuggestions = getTitleSuggestions(query);
        response.setTitleSuggestions(titleSuggestions);

        // Get category suggestions
        List<String> categorySuggestions = getCategorySuggestions(query);
        response.setCategorySuggestions(categorySuggestions);

        // Get tag suggestions
        List<String> tagSuggestions = getTagSuggestions(query);
        response.setTagSuggestions(tagSuggestions);

        // Get author suggestions
        List<String> authorSuggestions = getAuthorSuggestions(query);
        response.setAuthorSuggestions(authorSuggestions);

        return response;
    }

    /**
     * Get trending search terms
     */
    public List<String> getTrendingSearchTerms() {
        // In a real implementation, you would track search queries and return trending ones
        // For now, return popular categories and tags
        Aggregation aggregation = Aggregation.newAggregation(
            Aggregation.unwind("tags"),
            Aggregation.group("tags").count().as("count"),
            Aggregation.sort(Sort.Direction.DESC, "count"),
            Aggregation.limit(10)
        );

        AggregationResults<Map> results = mongoTemplate.aggregate(aggregation, "posts", Map.class);
        return results.getMappedResults().stream()
            .map(result -> (String) result.get("_id"))
            .collect(Collectors.toList());
    }

    /**
     * Get related posts based on a post ID
     */
    public List<PostResponse> getRelatedPosts(String postId, int limit) {
        PostEntity post = mongoTemplate.findById(postId, PostEntity.class);
        if (post == null) {
            return Collections.emptyList();
        }

        Query query = new Query();
        List<Criteria> criteria = new ArrayList<>();

        // Match by category
        if (post.getCategory() != null) {
            criteria.add(Criteria.where("category").is(post.getCategory()));
        }

        // Match by tags
        if (post.getTags() != null && !post.getTags().isEmpty()) {
            criteria.add(Criteria.where("tags").in(post.getTags()));
        }

        // Exclude the original post
        query.addCriteria(Criteria.where("id").ne(postId));

        if (!criteria.isEmpty()) {
            query.addCriteria(new Criteria().orOperator(criteria.toArray(new Criteria[0])));
        }

        query.with(Sort.by(Sort.Direction.DESC, "stats.likes", "stats.views"))
             .limit(limit);

        List<PostEntity> relatedPosts = mongoTemplate.find(query, PostEntity.class);
        return relatedPosts.stream()
            .map(PostResponse::new)
            .collect(Collectors.toList());
    }

    // Private helper methods

    private Query buildSearchQuery(SearchRequest searchRequest) {
        Query query = new Query();
        List<Criteria> criteria = new ArrayList<>();

        // Text search in title and content
        if (searchRequest.getQuery() != null && !searchRequest.getQuery().trim().isEmpty()) {
            String searchText = searchRequest.getQuery().trim();
            Criteria textCriteria = new Criteria().orOperator(
                Criteria.where("title").regex(searchText, "i"),
                Criteria.where("content").regex(searchText, "i")
            );
            criteria.add(textCriteria);
        }

        // Category filter
        if (searchRequest.getCategory() != null && !searchRequest.getCategory().trim().isEmpty()) {
            criteria.add(Criteria.where("category").is(searchRequest.getCategory()));
        }

        // Author filter
        if (searchRequest.getAuthorId() != null && !searchRequest.getAuthorId().trim().isEmpty()) {
            criteria.add(Criteria.where("author.id").is(searchRequest.getAuthorId()));
        }

        // Tags filter
        if (searchRequest.getTags() != null && !searchRequest.getTags().isEmpty()) {
            criteria.add(Criteria.where("tags").in(searchRequest.getTags()));
        }

        // Date range filter
        if (searchRequest.getDateFrom() != null || searchRequest.getDateTo() != null) {
            Criteria dateCriteria = Criteria.where("createdAt");
            if (searchRequest.getDateFrom() != null) {
                dateCriteria = dateCriteria.gte(searchRequest.getDateFrom());
            }
            if (searchRequest.getDateTo() != null) {
                dateCriteria = dateCriteria.lte(searchRequest.getDateTo());
            }
            criteria.add(dateCriteria);
        }

        // Privacy filter
        if (searchRequest.getVisibility() != null && !searchRequest.getVisibility().trim().isEmpty()) {
            criteria.add(Criteria.where("privacy").is(searchRequest.getVisibility()));
        } else {
            // Default to public posts if no specific visibility requested
            criteria.add(Criteria.where("privacy").is("PUBLIC"));
        }

        // Minimum engagement filters
        if (searchRequest.getMinLikes() != null) {
            criteria.add(Criteria.where("stats.likes").gte(searchRequest.getMinLikes()));
        }

        if (searchRequest.getMinViews() != null) {
            criteria.add(Criteria.where("stats.views").gte(searchRequest.getMinViews()));
        }

        if (searchRequest.getMinComments() != null) {
            criteria.add(Criteria.where("stats.comments").gte(searchRequest.getMinComments()));
        }

        // Post type filter
        if (searchRequest.getPostType() != null && !searchRequest.getPostType().trim().isEmpty()) {
            criteria.add(Criteria.where("postType").is(searchRequest.getPostType()));
        }

        // Combine all criteria
        if (!criteria.isEmpty()) {
            query.addCriteria(new Criteria().andOperator(criteria.toArray(new Criteria[0])));
        }

        return query;
    }

    private Sort buildSort(SearchRequest searchRequest) {
        String sortBy = searchRequest.getSortBy();
        String sortOrder = searchRequest.getSortOrder();

        Sort.Direction direction = "desc".equalsIgnoreCase(sortOrder) ?
            Sort.Direction.DESC : Sort.Direction.ASC;

        if ("date".equalsIgnoreCase(sortBy)) {
            return Sort.by(direction, "createdAt");
        } else if ("popularity".equalsIgnoreCase(sortBy)) {
            return Sort.by(Sort.Direction.DESC, "stats.likes", "stats.views", "stats.comments");
        } else if ("title".equalsIgnoreCase(sortBy)) {
            return Sort.by(direction, "title");
        } else {
            // Default relevance sorting (by engagement and recency)
            return Sort.by(Sort.Direction.DESC, "stats.likes", "createdAt");
        }
    }

    private Map<String, Object> buildFiltersApplied(SearchRequest searchRequest) {
        Map<String, Object> filters = new HashMap<>();

        if (searchRequest.getQuery() != null && !searchRequest.getQuery().trim().isEmpty()) {
            filters.put("query", searchRequest.getQuery().trim());
        }
        if (searchRequest.getCategory() != null && !searchRequest.getCategory().trim().isEmpty()) {
            filters.put("category", searchRequest.getCategory());
        }
        if (searchRequest.getAuthorId() != null && !searchRequest.getAuthorId().trim().isEmpty()) {
            filters.put("authorId", searchRequest.getAuthorId());
        }
        if (searchRequest.getTags() != null && !searchRequest.getTags().isEmpty()) {
            filters.put("tags", searchRequest.getTags());
        }
        if (searchRequest.getDateFrom() != null) {
            filters.put("dateFrom", searchRequest.getDateFrom().format(DateTimeFormatter.ISO_LOCAL_DATE));
        }
        if (searchRequest.getDateTo() != null) {
            filters.put("dateTo", searchRequest.getDateTo().format(DateTimeFormatter.ISO_LOCAL_DATE));
        }

        return filters;
    }

    // Helper methods for suggestions
    private List<String> getTitleSuggestions(String query) {
        Query mongoQuery = new Query();
        mongoQuery.addCriteria(Criteria.where("title").regex(query, "i"));
        mongoQuery.limit(5);

        List<PostEntity> posts = mongoTemplate.find(mongoQuery, PostEntity.class);
        return posts.stream()
                .map(PostEntity::getTitle)
                .distinct()
                .limit(5)
                .collect(Collectors.toList());
    }

    private List<String> getCategorySuggestions(String query) {
        Query mongoQuery = new Query();
        mongoQuery.addCriteria(Criteria.where("category").regex(query, "i"));

        List<PostEntity> posts = mongoTemplate.find(mongoQuery, PostEntity.class);
        return posts.stream()
                .map(PostEntity::getCategory)
                .filter(Objects::nonNull)
                .distinct()
                .limit(5)
                .collect(Collectors.toList());
    }

    private List<String> getTagSuggestions(String query) {
        Query mongoQuery = new Query();
        mongoQuery.addCriteria(Criteria.where("tags").regex(query, "i"));

        List<PostEntity> posts = mongoTemplate.find(mongoQuery, PostEntity.class);
        return posts.stream()
                .flatMap(post -> post.getTags().stream())
                .filter(tag -> tag.toLowerCase().contains(query.toLowerCase()))
                .distinct()
                .limit(5)
                .collect(Collectors.toList());
    }

    private List<String> getAuthorSuggestions(String query) {
        Query mongoQuery = new Query();
        mongoQuery.addCriteria(Criteria.where("author.name").regex(query, "i"));

        List<PostEntity> posts = mongoTemplate.find(mongoQuery, PostEntity.class);
        return posts.stream()
                .map(post -> post.getAuthor().getName())
                .filter(Objects::nonNull)
                .distinct()
                .limit(5)
                .collect(Collectors.toList());
    }
}
