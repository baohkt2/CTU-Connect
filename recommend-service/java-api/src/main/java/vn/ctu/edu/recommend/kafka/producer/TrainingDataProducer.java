package vn.ctu.edu.recommend.kafka.producer;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.model.dto.CandidatePost;
import vn.ctu.edu.recommend.model.dto.UserAcademicProfile;
import vn.ctu.edu.recommend.model.dto.UserInteractionHistory;

import java.util.HashMap;
import java.util.Map;

/**
 * Producer for sending training data to Python training pipeline
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class TrainingDataProducer {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    private static final String TRAINING_DATA_TOPIC = "recommendation_training_data";

    /**
     * Send complete training data sample to Python training pipeline
     * Format matches academic_dataset.json structure
     */
    public void sendTrainingDataSample(
            UserAcademicProfile userProfile,
            CandidatePost post,
            UserInteractionHistory interaction) {
        
        try {
            Map<String, Object> trainingData = new HashMap<>();
            trainingData.put("userProfile", userProfile);
            trainingData.put("post", post);
            trainingData.put("interaction", interaction);
            trainingData.put("timestamp", System.currentTimeMillis());

            kafkaTemplate.send(TRAINING_DATA_TOPIC, userProfile.getUserId(), trainingData);
            
            log.debug("Sent training data sample for user: {}, post: {}", 
                userProfile.getUserId(), post.getPostId());

        } catch (Exception e) {
            log.error("Error sending training data: {}", e.getMessage(), e);
        }
    }
}
