package vn.ctu.edu.recommend.ranking;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import vn.ctu.edu.recommend.model.dto.RecommendationResponse.RecommendedPost;
import vn.ctu.edu.recommend.model.entity.postgres.PostEmbedding;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Core ranking engine implementing the recommendation algorithm
 * 
 * final_score = α * content_similarity + β * graph_relation_score + 
 *               γ * academic_score + δ * popularity_score
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class RankingEngine {

    @Value("${recommendation.weights.content-similarity}")
    private float alphaWeight;  // α

    @Value("${recommendation.weights.graph-relation}")
    private float betaWeight;   // β

    @Value("${recommendation.weights.academic-score}")
    private float gammaWeight;  // γ

    @Value("${recommendation.weights.popularity-score}")
    private float deltaWeight;  // δ

    /**
     * Calculate final recommendation score for a post
     * 
     * @param contentSimilarity Cosine similarity from PhoBERT embeddings (0-1)
     * @param graphRelationScore Neo4j graph relationship score (0-1)
     * @param academicScore Academic classification confidence (0-1)
     * @param popularityScore Post engagement score (0-1)
     * @return Final weighted score (0-1)
     */
    public float computeFinalScore(
            float contentSimilarity,
            float graphRelationScore,
            float academicScore,
            float popularityScore) {
        
        // Normalize inputs to 0-1 range
        contentSimilarity = normalize(contentSimilarity, 0, 1);
        graphRelationScore = normalize(graphRelationScore, 0, 1);
        academicScore = normalize(academicScore, 0, 1);
        popularityScore = normalize(popularityScore, 0, 1);

        // Weighted sum
        float finalScore = 
            alphaWeight * contentSimilarity +
            betaWeight * graphRelationScore +
            gammaWeight * academicScore +
            deltaWeight * popularityScore;

        log.debug("Score computed - Content: {}, Graph: {}, Academic: {}, Popularity: {} -> Final: {}",
            contentSimilarity, graphRelationScore, academicScore, popularityScore, finalScore);

        return finalScore;
    }

    /**
     * Rank a list of posts based on the recommendation algorithm
     */
    public List<RecommendedPost> rankPosts(
            List<PostEmbedding> posts,
            Map<String, Float> contentSimilarities,
            Map<String, Float> graphRelationScores,
            int limit) {

        List<RecommendedPost> rankedPosts = new ArrayList<>();

        for (PostEmbedding post : posts) {
            String postId = post.getPostId();
            
            // Get scores from maps or use defaults
            float contentSimilarity = contentSimilarities.getOrDefault(postId, 0.0f);
            float graphRelationScore = graphRelationScores.getOrDefault(postId, 0.0f);
            float academicScore = post.getAcademicScore() != null ? post.getAcademicScore() : 0.0f;
            float popularityScore = normalizePopularityScore(post.getPopularityScore());

            // Calculate final score
            float finalScore = computeFinalScore(
                contentSimilarity,
                graphRelationScore,
                academicScore,
                popularityScore
            );

            // Build recommended post
            RecommendedPost recommendedPost = RecommendedPost.builder()
                .postId(postId)
                .authorId(post.getAuthorId())
                .content(post.getContent())
                .finalScore(finalScore)
                .contentSimilarity(contentSimilarity)
                .graphRelationScore(graphRelationScore)
                .academicScore(academicScore)
                .popularityScore(popularityScore)
                .academicCategory(post.getAcademicCategory())
                .build();

            rankedPosts.add(recommendedPost);
        }

        // Sort by final score descending
        rankedPosts.sort(Comparator.comparing(RecommendedPost::getFinalScore).reversed());

        // Assign ranks
        for (int i = 0; i < rankedPosts.size(); i++) {
            rankedPosts.get(i).setRank(i + 1);
        }

        // Apply diversity and limit
        List<RecommendedPost> diversified = applyDiversity(rankedPosts);
        
        return diversified.stream()
            .limit(limit)
            .collect(Collectors.toList());
    }

    /**
     * Apply diversity to avoid echo chambers
     * Ensures variety in academic categories and authors
     */
    private List<RecommendedPost> applyDiversity(List<RecommendedPost> posts) {
        if (posts.size() <= 10) {
            return posts; // Too small to diversify
        }

        List<RecommendedPost> diversified = new ArrayList<>();
        Set<String> seenAuthors = new HashSet<>();
        Set<String> seenCategories = new HashSet<>();
        
        int maxPostsPerAuthor = Math.max(3, posts.size() / 10);
        int maxPostsPerCategory = Math.max(5, posts.size() / 5);

        for (RecommendedPost post : posts) {
            String authorId = post.getAuthorId();
            String category = post.getAcademicCategory();

            // Count occurrences
            long authorCount = diversified.stream()
                .filter(p -> p.getAuthorId().equals(authorId))
                .count();
            long categoryCount = diversified.stream()
                .filter(p -> Objects.equals(p.getAcademicCategory(), category))
                .count();

            // Apply diversity constraints
            if (authorCount < maxPostsPerAuthor && 
                (category == null || categoryCount < maxPostsPerCategory)) {
                diversified.add(post);
                seenAuthors.add(authorId);
                if (category != null) {
                    seenCategories.add(category);
                }
            }
        }

        // Fill remaining slots if needed
        if (diversified.size() < posts.size()) {
            posts.stream()
                .filter(p -> !diversified.contains(p))
                .limit(posts.size() - diversified.size())
                .forEach(diversified::add);
        }

        return diversified;
    }

    /**
     * Normalize popularity score to 0-1 range using logarithmic scaling
     */
    private float normalizePopularityScore(Float rawScore) {
        if (rawScore == null || rawScore <= 0) {
            return 0.0f;
        }
        
        // Log scaling to handle outliers
        float logScore = (float) Math.log1p(rawScore);
        // Assume max reasonable popularity score is ~1000 (log ~7)
        float maxLogScore = (float) Math.log1p(1000);
        
        return Math.min(1.0f, logScore / maxLogScore);
    }

    /**
     * Normalize value to specified range
     */
    private float normalize(float value, float min, float max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    /**
     * Calculate time decay factor for recency
     */
    public float calculateTimeDecay(long ageInHours) {
        // Exponential decay: score = e^(-λt)
        // Half-life of 24 hours (λ = ln(2)/24)
        double lambda = Math.log(2) / 24.0;
        return (float) Math.exp(-lambda * ageInHours);
    }

    /**
     * Apply personalization boost based on user preferences
     */
    public float applyPersonalizationBoost(
            RecommendedPost post,
            Set<String> userInterests,
            String userFaculty,
            String userMajor) {
        
        float boost = 1.0f;

        // Faculty match
        if (userFaculty != null && userFaculty.equals(post.getAuthorId())) {
            boost += 0.1f;
        }

        // Category interest match
        if (post.getAcademicCategory() != null && 
            userInterests.contains(post.getAcademicCategory().toLowerCase())) {
            boost += 0.15f;
        }

        return Math.min(boost, 1.5f); // Cap boost at 50%
    }

    /**
     * Get current weight configuration
     */
    public Map<String, Float> getWeights() {
        Map<String, Float> weights = new HashMap<>();
        weights.put("alpha_content_similarity", alphaWeight);
        weights.put("beta_graph_relation", betaWeight);
        weights.put("gamma_academic_score", gammaWeight);
        weights.put("delta_popularity_score", deltaWeight);
        return weights;
    }

    /**
     * Validate that weights sum to approximately 1.0
     */
    public boolean validateWeights() {
        float sum = alphaWeight + betaWeight + gammaWeight + deltaWeight;
        float epsilon = 0.01f;
        boolean valid = Math.abs(sum - 1.0f) < epsilon;
        
        if (!valid) {
            log.warn("Weight sum is {}, expected 1.0", sum);
        }
        
        return valid;
    }
}
