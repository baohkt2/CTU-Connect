package com.ctuconnect.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.convert.ReadingConverter;
import org.springframework.data.convert.WritingConverter;
import org.springframework.data.neo4j.config.EnableNeo4jAuditing;
import org.springframework.data.neo4j.core.DatabaseSelectionProvider;

import org.springframework.data.neo4j.core.convert.Neo4jConversions;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.neo4j.cypherdsl.core.renderer.Dialect;
import org.springframework.core.convert.converter.Converter;

import org.neo4j.driver.Value;


import java.time.LocalDateTime;
import java.time.ZonedDateTime;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;

@Configuration
@EnableNeo4jAuditing
@EnableTransactionManagement
public class Neo4jConfig {

    @Bean
    public DatabaseSelectionProvider databaseSelectionProvider() {
        return DatabaseSelectionProvider.getDefaultSelectionProvider();
    }
    
    /**
     * Configure Cypher DSL to use Neo4j 5.x syntax
     * This fixes the deprecated id() function warning by using elementId() instead
     */
    @Bean
    org.neo4j.cypherdsl.core.renderer.Configuration cypherDslConfiguration() {
        return org.neo4j.cypherdsl.core.renderer.Configuration.newConfig()
            .withDialect(Dialect.NEO4J_5)
            .build();
    }
    
    /**
     * Custom converters for handling Neo4j DateTime to Java LocalDateTime
     */
    @Bean
    public Neo4jConversions neo4jConversions() {
        List<Converter<?, ?>> converters = new ArrayList<>();
        converters.add(new Neo4jDateTimeValueToLocalDateTimeConverter());
        converters.add(new ZonedDateTimeToLocalDateTimeConverter());
        converters.add(new InstantToLocalDateTimeConverter());
        converters.add(new StringToLocalDateTimeConverter());

        // BỔ SUNG CONVERTER TỪ INTEGER/LONG SANG STRING VÀ NGƯỢC LẠI
        converters.add(new IntegerToStringConverter());
        converters.add(new StringToIntegerConverter());
        return new Neo4jConversions(converters);
    }
         // --- CONVERTERS CHO INTEGER/LONG <-> STRING ---
    /**
     * Convert Integer/Long to String (thường dùng khi DB lưu số, Java Entity lưu String)
     */
    @ReadingConverter
    public static class IntegerToStringConverter implements Converter<Number, String> {
        @Override
        public String convert(Number source) {
            return source == null ? null : source.toString();
        }
    }

    /**
     * Convert String to Integer (thường dùng khi DB lưu String, Java Entity lưu Integer)
     */
    @WritingConverter
    public static class StringToIntegerConverter implements Converter<String, Integer> {
        @Override
        public Integer convert(String source) {
            try {
                return source == null || source.isEmpty() ? null : Integer.valueOf(source);
            } catch (NumberFormatException e) {
                // Tùy chọn: Log lỗi hoặc ném ngoại lệ nếu chuỗi không phải là số hợp lệ
                throw new IllegalArgumentException("Cannot convert non-numeric String to Integer: " + source, e);
            }
        }
    } 
    /**
     * Convert Neo4j DateTimeValue to LocalDateTime
     * This is the main converter for Neo4j driver's native DateTime type
     */
    public static class Neo4jDateTimeValueToLocalDateTimeConverter implements Converter<Value, LocalDateTime> {
        @Override
        public LocalDateTime convert(Value source) {
            if (source == null || source.isNull()) {
                return null;
            }
            
            try {
                // Neo4j DateTimeValue can be converted to ZonedDateTime
                ZonedDateTime zonedDateTime = source.asZonedDateTime();
                return zonedDateTime.withZoneSameInstant(ZoneOffset.UTC).toLocalDateTime();
            } catch (Exception e) {
                try {
                    // Try as LocalDateTime directly
                    return source.asLocalDateTime();
                } catch (Exception ex) {
                    // Last resort: try as Instant
                    return LocalDateTime.ofInstant(Instant.ofEpochMilli(source.asLong()), ZoneOffset.UTC);
                }
            }
        }
    }
    
    /**
     * Convert ZonedDateTime from Neo4j to LocalDateTime
     */
    public static class ZonedDateTimeToLocalDateTimeConverter implements Converter<ZonedDateTime, LocalDateTime> {
        @Override
        public LocalDateTime convert(ZonedDateTime source) {
            return source.withZoneSameInstant(ZoneOffset.UTC).toLocalDateTime();
        }
    }
    
    /**
     * Convert Instant from Neo4j to LocalDateTime
     */
    public static class InstantToLocalDateTimeConverter implements Converter<Instant, LocalDateTime> {
        @Override
        public LocalDateTime convert(Instant source) {
            return LocalDateTime.ofInstant(source, ZoneOffset.UTC);
        }
    }
    
    /**
     * Convert ISO8601 String from Neo4j to LocalDateTime
     */
    public static class StringToLocalDateTimeConverter implements Converter<String, LocalDateTime> {
        @Override
        public LocalDateTime convert(String source) {
            try {
                // Try parsing as ZonedDateTime first (e.g., "2025-12-04T10:57:28.342Z")
                return ZonedDateTime.parse(source).withZoneSameInstant(ZoneOffset.UTC).toLocalDateTime();
            } catch (Exception e) {
                try {
                    // Try parsing as Instant
                    return LocalDateTime.ofInstant(Instant.parse(source), ZoneOffset.UTC);
                } catch (Exception ex) {
                    // Fallback to direct LocalDateTime parsing
                    return LocalDateTime.parse(source);
                }
            }
        }
        
    }
    
}

