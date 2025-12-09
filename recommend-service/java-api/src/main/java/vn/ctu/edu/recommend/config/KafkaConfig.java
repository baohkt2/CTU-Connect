package vn.ctu.edu.recommend.config;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

/**
 * Kafka configuration for topics only
 * Consumer configurations are in KafkaConsumerConfig
 */
@Configuration
public class KafkaConfig {

    @Bean
    public NewTopic postCreatedTopic() {
        return TopicBuilder.name("post_created")
            .partitions(3)
            .replicas(1)
            .build();
    }

    @Bean
    public NewTopic postUpdatedTopic() {
        return TopicBuilder.name("post_updated")
            .partitions(3)
            .replicas(1)
            .build();
    }

    @Bean
    public NewTopic postDeletedTopic() {
        return TopicBuilder.name("post_deleted")
            .partitions(3)
            .replicas(1)
            .build();
    }

    @Bean
    public NewTopic userActionTopic() {
        return TopicBuilder.name("user_action")
            .partitions(3)
            .replicas(1)
            .build();
    }
    
    @Bean
    public NewTopic commentEventsTopic() {
        return TopicBuilder.name("comment-events")
            .partitions(3)
            .replicas(1)
            .build();
    }
}
