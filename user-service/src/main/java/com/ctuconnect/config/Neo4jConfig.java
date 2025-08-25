package com.ctuconnect.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.neo4j.config.EnableNeo4jAuditing;
import org.springframework.data.neo4j.core.DatabaseSelectionProvider;
import org.springframework.data.neo4j.core.transaction.Neo4jTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@Configuration
@EnableNeo4jAuditing
@EnableTransactionManagement
public class Neo4jConfig {

    @Bean
    public DatabaseSelectionProvider databaseSelectionProvider() {
        return DatabaseSelectionProvider.getDefaultSelectionProvider();
    }
}
