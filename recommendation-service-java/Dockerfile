FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /app

# Copy pom.xml and download dependencies
COPY pom.xml ./
RUN mvn dependency:go-offline -B

# Copy source code
COPY src ./src

# Build application
RUN mvn clean package -DskipTests

# Runtime stage
FROM eclipse-temurin:17-jdk-focal
WORKDIR /app

# Copy built artifact from build stage
COPY --from=build /app/target/*.jar app.jar

# Expose port
EXPOSE 8095

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8095/actuator/health || exit 1

# Run application
ENTRYPOINT ["java", "-jar", "app.jar"]
