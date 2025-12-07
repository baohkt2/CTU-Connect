package vn.ctu.edu.recommend.nlp;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import vn.ctu.edu.recommend.model.dto.ClassificationRequest;
import vn.ctu.edu.recommend.model.dto.ClassificationResponse;
import vn.ctu.edu.recommend.model.enums.AcademicCategory;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Service for classifying academic content
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class AcademicClassifier {

    private final WebClient.Builder webClientBuilder;

    @Value("${recommendation.nlp.phobert-service-url}")
    private String phoBertServiceUrl;

    @Value("${recommendation.nlp.classifier-endpoint}")
    private String classifierEndpoint;

    @Value("${recommendation.nlp.timeout}")
    private long timeout;

    @Value("${recommendation.academic.min-confidence}")
    private float minConfidence;

    // Academic keywords for fallback classification
    private static final Map<AcademicCategory, String[]> ACADEMIC_KEYWORDS = new HashMap<>();
    
    static {
        ACADEMIC_KEYWORDS.put(AcademicCategory.RESEARCH, new String[]{
            "nghiên cứu", "research", "paper", "journal", "publication", "phát hiện", 
            "kết quả", "phương pháp", "methodology", "analysis", "phân tích"
        });
        ACADEMIC_KEYWORDS.put(AcademicCategory.SCHOLARSHIP, new String[]{
            "học bổng", "scholarship", "tài trợ", "funding", "grant", "hỗ trợ", 
            "miễn giảm", "stipend"
        });
        ACADEMIC_KEYWORDS.put(AcademicCategory.QA, new String[]{
            "hỏi đáp", "câu hỏi", "question", "answer", "giải đáp", "thắc mắc", 
            "tư vấn", "advice", "help"
        });
        ACADEMIC_KEYWORDS.put(AcademicCategory.ANNOUNCEMENT, new String[]{
            "thông báo", "announcement", "notice", "công bố", "thông tin", 
            "notification", "update"
        });
        ACADEMIC_KEYWORDS.put(AcademicCategory.EVENT, new String[]{
            "sự kiện", "event", "hội thảo", "seminar", "workshop", "conference", 
            "đại hội", "diễn đàn", "forum", "tọa đàm"
        });
        ACADEMIC_KEYWORDS.put(AcademicCategory.COURSE, new String[]{
            "khóa học", "course", "môn học", "subject", "giáo trình", "curriculum", 
            "chương trình", "syllabus", "bài giảng", "lecture"
        });
        ACADEMIC_KEYWORDS.put(AcademicCategory.PROJECT, new String[]{
            "dự án", "project", "đồ án", "assignment", "bài tập", "exercise", 
            "thực hành", "practice", "lab"
        });
        ACADEMIC_KEYWORDS.put(AcademicCategory.THESIS, new String[]{
            "luận văn", "thesis", "luận án", "dissertation", "khóa luận", 
            "graduation project", "tốt nghiệp"
        });
    }

    /**
     * Classify text as academic or non-academic with category
     */
    public ClassificationResponse classify(String text) {
        try {
            // Try ML-based classification first
            ClassificationRequest request = ClassificationRequest.builder()
                .text(text)
                .model("academic-classifier")
                .build();

            WebClient webClient = webClientBuilder.baseUrl(phoBertServiceUrl).build();
            
            ClassificationResponse response = webClient.post()
                .uri(classifierEndpoint)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(ClassificationResponse.class)
                .timeout(Duration.ofMillis(timeout))
                .onErrorResume(e -> {
                    log.warn("ML classifier unavailable, using fallback: {}", e.getMessage());
                    return Mono.just(fallbackClassify(text));
                })
                .block();

            if (response != null && response.getConfidence() != null) {
                log.debug("Classified text as: {} (confidence: {})", 
                    response.getCategory(), response.getConfidence());
                return response;
            }

            // Fallback to rule-based
            return fallbackClassify(text);

        } catch (Exception e) {
            log.error("Error classifying text: {}", e.getMessage(), e);
            return fallbackClassify(text);
        }
    }

    /**
     * Rule-based fallback classification
     */
    private ClassificationResponse fallbackClassify(String text) {
        if (text == null || text.trim().isEmpty()) {
            return createNonAcademicResponse();
        }

        String lowerText = text.toLowerCase();
        
        // Calculate scores for each category
        Map<AcademicCategory, Float> scores = new HashMap<>();
        float maxScore = 0.0f;
        AcademicCategory bestCategory = AcademicCategory.NON_ACADEMIC;

        for (Map.Entry<AcademicCategory, String[]> entry : ACADEMIC_KEYWORDS.entrySet()) {
            float score = calculateKeywordScore(lowerText, entry.getValue());
            scores.put(entry.getKey(), score);
            
            if (score > maxScore) {
                maxScore = score;
                bestCategory = entry.getKey();
            }
        }

        // Convert scores to probabilities
        Map<String, Float> probabilities = new HashMap<>();
        for (Map.Entry<AcademicCategory, Float> entry : scores.entrySet()) {
            probabilities.put(entry.getKey().name(), entry.getValue());
        }

        boolean isAcademic = maxScore >= minConfidence;
        
        return ClassificationResponse.builder()
            .category(isAcademic ? bestCategory.name() : AcademicCategory.NON_ACADEMIC.name())
            .confidence(maxScore)
            .isAcademic(isAcademic)
            .probabilities(probabilities)
            .build();
    }

    /**
     * Calculate keyword match score
     */
    private float calculateKeywordScore(String text, String[] keywords) {
        int matchCount = 0;
        for (String keyword : keywords) {
            if (Pattern.compile("\\b" + Pattern.quote(keyword) + "\\b", Pattern.CASE_INSENSITIVE)
                    .matcher(text).find()) {
                matchCount++;
            }
        }
        
        // Normalize score (0-1)
        float score = (float) matchCount / keywords.length;
        
        // Boost if multiple keywords match
        if (matchCount > 1) {
            score = Math.min(1.0f, score * 1.5f);
        }
        
        return score;
    }

    /**
     * Create non-academic classification response
     */
    private ClassificationResponse createNonAcademicResponse() {
        return ClassificationResponse.builder()
            .category(AcademicCategory.NON_ACADEMIC.name())
            .confidence(1.0f)
            .isAcademic(false)
            .probabilities(new HashMap<>())
            .build();
    }

    /**
     * Quick check if text contains academic indicators
     */
    public boolean isLikelyAcademic(String text) {
        if (text == null || text.trim().isEmpty()) {
            return false;
        }

        String lowerText = text.toLowerCase();
        
        // Check for any academic keywords
        for (String[] keywords : ACADEMIC_KEYWORDS.values()) {
            for (String keyword : keywords) {
                if (Pattern.compile("\\b" + Pattern.quote(keyword) + "\\b", Pattern.CASE_INSENSITIVE)
                        .matcher(lowerText).find()) {
                    return true;
                }
            }
        }
        
        return false;
    }
}
