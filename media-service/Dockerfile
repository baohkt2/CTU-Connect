# ------------ Stage 1: Build Application ----------------
FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /app
# Copy source code
COPY pom.xml .
COPY src ./src

# Package app (skip tests to speed up if needed)
RUN mvn clean package -DskipTests

# ------------ Stage 2: Create Runtime Image ------------
FROM eclipse-temurin:17-jdk-focal

WORKDIR /app

# Copy the built jar from previous stage
COPY --from=build /app/target/media-service-0.0.1-SNAPSHOT.jar app.jar

# Expose application port
EXPOSE 8084

# Environment variables
ENV SPRING_PROFILES_ACTIVE=docker

# Run the app
ENTRYPOINT ["java", "-jar", "app.jar"]
