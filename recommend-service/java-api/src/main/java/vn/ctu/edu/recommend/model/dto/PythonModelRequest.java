package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

/**
 * Request to Python ML model service for ranking
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PythonModelRequest implements Serializable {
    private UserAcademicProfile userAcademic;
    private List<UserInteractionHistory> userHistory;
    private List<CandidatePost> candidatePosts;
    private Integer topK;  // Number of recommendations to return
}
