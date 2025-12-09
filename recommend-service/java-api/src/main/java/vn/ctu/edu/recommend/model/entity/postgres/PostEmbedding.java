package vn.ctu.edu.recommend.model.entity.postgres;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * Post embedding entity with vector storage using pgvector
 */
@Entity
@Table(name = "post_embeddings", indexes = {
    @Index(name = "idx_post_id", columnList = "post_id"),
    @Index(name = "idx_author_id", columnList = "author_id"),
    @Index(name = "idx_created_at", columnList = "created_at"),
    @Index(name = "idx_academic_score", columnList = "academic_score")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PostEmbedding {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "post_id", nullable = false, unique = true)
    private String postId;

    @Column(name = "author_id", nullable = false)
    private String authorId;

    @Column(name = "content", columnDefinition = "TEXT")
    private String content;

    /**
     * PhoBERT embedding vector (768 dimensions)
     * Stored as TEXT in pgvector format
     */
    @Column(name = "embedding_vector", columnDefinition = "TEXT")
    private String embeddingVector;

    /**
     * Academic classification score (0-1)
     * Higher score means more academic content
     */
    @Column(name = "academic_score", nullable = false)
    private Float academicScore = 0.0f;

    /**
     * Academic category classification
     */
    @Column(name = "academic_category", length = 50)
    private String academicCategory;

    /**
     * Popularity score calculated from engagements
     */
    @Column(name = "popularity_score", nullable = false)
    private Float popularityScore = 0.0f;
    
    /**
     * Content similarity score (for recommendations)
     */
    @Column(name = "content_similarity_score")
    private Float contentSimilarityScore = 0.0f;
    
    /**
     * Graph relation score (social connections)
     */
    @Column(name = "graph_relation_score")
    private Float graphRelationScore = 0.0f;

    /**
     * Engagement metrics
     */
    @Column(name = "like_count", nullable = false)
    private Integer likeCount = 0;

    @Column(name = "comment_count", nullable = false)
    private Integer commentCount = 0;

    @Column(name = "share_count", nullable = false)
    private Integer shareCount = 0;

    @Column(name = "view_count", nullable = false)
    private Integer viewCount = 0;

    /**
     * Metadata
     */
    @Column(name = "faculty", length = 100)
    private String faculty;

    @Column(name = "major", length = 100)
    private String major;

    @Column(name = "author_major", length = 100)
    private String authorMajor;

    @Column(name = "author_faculty", length = 100)
    private String authorFaculty;

    @Column(name = "media_description", columnDefinition = "TEXT")
    private String mediaDescription;

    @Column(name = "tags", columnDefinition = "TEXT[]")
    private String[] tags;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @Column(name = "embedding_updated_at")
    private LocalDateTime embeddingUpdatedAt;

    /**
     * Set embedding vector from float array
     */
    public void setEmbeddingVectorFromArray(float[] vector) {
        if (vector != null) {
            // Convert to pgvector format string: "[1.0,2.0,3.0,...]"
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < vector.length; i++) {
                if (i > 0) sb.append(",");
                sb.append(vector[i]);
            }
            sb.append("]");
            this.embeddingVector = sb.toString();
        }
    }

    /**
     * Get embedding vector as float array
     */
    public float[] getEmbeddingVectorAsArray() {
        if (embeddingVector != null && embeddingVector.startsWith("[")) {
            String content = embeddingVector.substring(1, embeddingVector.length() - 1);
            String[] values = content.split(",");
            float[] result = new float[values.length];
            for (int i = 0; i < values.length; i++) {
                result[i] = Float.parseFloat(values[i].trim());
            }
            return result;
        }
        return null;
    }

    /**
     * Calculate and update popularity score
     */
    public void calculatePopularityScore() {
        // Weighted popularity formula
        this.popularityScore = (float) (
            0.4 * likeCount +
            0.3 * commentCount +
            0.2 * shareCount +
            0.1 * Math.log1p(viewCount)
        );
    }
}
