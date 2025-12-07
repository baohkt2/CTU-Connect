package vn.ctu.edu.recommend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.data.neo4j.repository.config.EnableNeo4jRepositories;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * CTU Connect Advanced Recommendation Service
 * 
 * Main Application class for AI-powered personalized recommendation system
 * Features:
 * - Content-based filtering with PhoBERT Vietnamese embeddings
 * - Collaborative filtering using Neo4j graph relationships
 * - Academic content classification
 * - Popularity-based ranking
 * - Real-time event processing with Kafka
 * - Multi-level caching with Redis
 * 
 * @author CTU Connect Team
 * @version 1.0.0
 */
@SpringBootApplication
@EnableDiscoveryClient
@EnableFeignClients
@EnableKafka
@EnableCaching
@EnableAsync
@EnableScheduling
@EnableJpaRepositories(basePackages = "vn.ctu.edu.recommend.repository.postgres")
@EnableNeo4jRepositories(basePackages = "vn.ctu.edu.recommend.repository.neo4j")
public class RecommendationServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(RecommendationServiceApplication.class, args);
    }
}
