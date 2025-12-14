package vn.ctu.edu.recommend.model.entity.postgres;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

/**
 * User embedding entity for friend recommendation
 * Stores PhoBERT embeddings of user profiles
 */
@Entity
@Table(name = "user_embeddings", schema = "recommend", indexes = {
    @Index(name = "idx_user_emb_user_id", columnList = "user_id"),
    @Index(name = "idx_user_emb_faculty", columnList = "faculty"),
    @Index(name = "idx_user_emb_major", columnList = "major"),
    @Index(name = "idx_user_emb_updated_at", columnList = "updated_at")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserEmbedding {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Integer id;

    @Column(name = "user_id", nullable = false, unique = true)
    private String userId;

    /**
     * PhoBERT embedding vector (768 dimensions)
     * Stored as FLOAT[] in PostgreSQL
     */
    @Column(name = "embedding", columnDefinition = "FLOAT[]", nullable = false)
    private float[] embedding;

    @Column(name = "dimension", nullable = false)
    private Integer dimension;

    @Column(name = "major", length = 100)
    private String major;

    @Column(name = "faculty", length = 100)
    private String faculty;

    @Column(name = "bio", columnDefinition = "TEXT")
    private String bio;

    @Column(name = "batch_year", length = 20)
    private String batchYear;

    /**
     * User interests as array
     */
    @Column(name = "interests", columnDefinition = "TEXT[]")
    private String[] interests;

    /**
     * User skills as array
     */
    @Column(name = "skills", columnDefinition = "TEXT[]")
    private String[] skills;

    /**
     * Additional metadata in JSON format
     */
    @Column(name = "metadata", columnDefinition = "JSONB")
    private String metadata;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    /**
     * Convert embedding array to string for vector operations
     */
    public String getEmbeddingAsString() {
        if (embedding == null) return null;
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < embedding.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(embedding[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Set embedding from list
     */
    public void setEmbeddingFromList(List<Float> embeddingList) {
        if (embeddingList == null) {
            this.embedding = null;
            this.dimension = null;
            return;
        }
        this.embedding = new float[embeddingList.size()];
        for (int i = 0; i < embeddingList.size(); i++) {
            this.embedding[i] = embeddingList.get(i);
        }
        this.dimension = embeddingList.size();
    }
}
