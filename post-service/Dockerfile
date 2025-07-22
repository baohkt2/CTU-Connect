# ------------ Stage 1: Build Application ----------------
FROM maven:3.9.4-eclipse-temurin-17 AS builder

WORKDIR /build

COPY pom.xml .
COPY src ./src

RUN mvn clean package -DskipTests

# ------------ Stage 2: Create Runtime Image ------------
FROM openjdk:17-jdk-slim

WORKDIR /app

COPY --from=builder /build/target/post-service-0.0.1-SNAPSHOT.jar app.jar

EXPOSE 8085

ENV SPRING_PROFILES_ACTIVE=docker

ENTRYPOINT ["java", "-jar", "app.jar"]
