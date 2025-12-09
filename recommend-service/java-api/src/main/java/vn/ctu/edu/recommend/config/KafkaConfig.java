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
    
    /**
     * Consumer factory for user_action topic with flexible deserialization
     * Accepts any Object (Map or typed object)
     */
    @Bean
    public ConsumerFactory<String, Object> userActionConsumerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "recommendation-service-group");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class);
        props.put(JsonDeserializer.TRUSTED_PACKAGES, "*");
        props.put(JsonDeserializer.USE_TYPE_INFO_HEADERS, false); // Don't use type headers
        props.put(JsonDeserializer.VALUE_DEFAULT_TYPE, "java.util.LinkedHashMap"); // Default to Map
        
        return new DefaultKafkaConsumerFactory<>(props);
    }
    
    /**
     * Listener container factory for user_action topic
     */
    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, Object> userActionKafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, Object> factory = 
            new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(userActionConsumerFactory());
        return factory;
    }
}
