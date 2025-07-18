# ------------ Stage 1: Build Application ----------------
FROM maven:3.9.4-eclipse-temurin-17 AS builder

WORKDIR /build

# Copy source code
COPY pom.xml .
COPY src ./src

# Package app (skip tests to speed up if needed)
RUN mvn clean package -DskipTests

# ------------ Stage 2: Create Runtime Image ------------
FROM openjdk:17-jdk-slim

WORKDIR /app

# Copy the built jar from previous stage
COPY --from=builder /build/target/media-service-0.0.1-SNAPSHOT.jar app.jar

# Expose application port
EXPOSE 8084

# Environment variables
ENV SPRING_PROFILES_ACTIVE=docker

# Run the app
ENTRYPOINT ["java", "-jar", "app.jar"]
