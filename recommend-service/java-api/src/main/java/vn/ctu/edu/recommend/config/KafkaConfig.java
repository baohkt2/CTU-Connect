package vn.ctu.edu.recommend.config;

import org.apache.kafka.clients.admin.NewTopic;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.kafka.config.TopicBuilder;
import org.springframework.kafka.core.ConsumerFactory;
import org.springframework.kafka.core.DefaultKafkaConsumerFactory;
import org.springframework.kafka.support.serializer.JsonDeserializer;

import java.util.HashMap;
import java.util.Map;

/**
 * Kafka configuration for topics and consumers
 */
@EnableKafka
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
}
