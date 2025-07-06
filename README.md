# CTU Connect Microservice Architecture
# CTU Microservices Application
# CTU Connect Project

## Overview
This project implements a microservices architecture using Spring Boot, Spring Cloud, and related technologies.

## Services
- **Eureka Server**: Service discovery server
- **Auth Service**: Authentication and user management service

## Infrastructure
- **PostgreSQL**: Database for storing application data
- **Redis**: Cache service for improved performance

## Getting Started

### Prerequisites
- Docker and Docker Compose installed
- Java 17 (for local development)
- Maven (for local development)

### Running with Docker Compose

1. Build and start all services:
   ```bash
   docker-compose up -d
   ```

2. Stop all services:
   ```bash
   docker-compose down
   ```

3. View logs:
   ```bash
   docker-compose logs -f [service-name]
   ```

4. Rebuild a specific service:
   ```bash
   docker-compose build [service-name]
   docker-compose up -d [service-name]
   ```

### Service URLs
- Eureka Server: http://localhost:8761
- Auth Service: http://localhost:8080

### Environment Variables
All required environment variables are defined in the `.env` file.

## Development

### Adding a New Service
1. Create a new Spring Boot project
2. Add Eureka client dependency
3. Configure the service in docker-compose.yml
4. Create a Dockerfile for the service

### Database Initialization
Database initialization scripts are located in the `postgres-init` directory.
This is a microservices-based application using Spring Boot, Spring Cloud, and other technologies.

## Services

- **Eureka Server**: Service discovery server (port 8761)
- **Auth Service**: Authentication and authorization service (port 8080)

## Infrastructure

- **PostgreSQL**: Database for auth service
- **Redis**: For caching and session management
- **Zookeeper & Kafka**: For event-driven communication between services

## Running the Application

1. Make sure you have Docker and Docker Compose installed
2. Clone this repository
3. Run the application with:

```bash
docker-compose up -d
```

4. To check service health and logs:

```bash
# View all running containers
docker-compose ps

# Check logs for a specific service
docker-compose logs -f auth-service
```

5. Access services:
   - Eureka Dashboard: http://localhost:8761
   - Auth Service: http://localhost:8080

## Stopping the Application

```bash
docker-compose down
```

To remove volumes as well (this will delete persistent data):

```bash
docker-compose down -v
```

## Configuration

Environment variables are stored in the `.env` file. Update this file to change configurations.

## Troubleshooting

If services fail to start, check the logs for each service:

```bash
docker-compose logs -f <service-name>
```

Make sure all dependencies are running correctly with health checks:

```bash
docker ps
```

Look for health status in the output.
## Project Overview
This project implements a microservice architecture for CTU Connect, consisting of multiple services:

- **Eureka Server**: Service discovery server
- **Auth Service**: Handles user authentication and email verification
- **User Service**: Manages user profiles and data
- **Recommendation Service**: Provides recommendation functionality

## Prerequisites

- Docker and Docker Compose
- Java 17 (for local development)
- Maven (for local development)

## Getting Started

### Running the Application with Docker Compose

1. Clone the repository

2. Start all services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. To view logs for a specific service:
   ```bash
   docker-compose logs -f service-name
   ```

4. To stop all services:
   ```bash
   docker-compose down
   ```

### Development Workflow

#### Building Individual Services

If you want to build a service locally:

```bash
cd service-name
./mvnw clean package
```

#### Running Individual Services Locally

To run a service locally during development:

```bash
cd service-name
./mvnw spring-boot:run
```

## Service URLs

- **Eureka Server**: http://localhost:8761
- **Auth Service**: http://localhost:8080
- **User Service**: http://localhost:8081
- **Recommendation Service**: http://localhost:8082

## Troubleshooting

### Common Issues

1. **Service dependencies**: Services have dependencies on each other. Make sure the dependent services are running before starting a service that depends on them.

2. **Database initialization**: If you're experiencing database connection issues, check if the database containers are healthy using `docker-compose ps`.

3. **Building issues**: If you encounter problems with the multi-stage Docker builds, you can build the JAR files locally and modify the Dockerfile to copy the local JAR file instead.

## Additional Notes

- The configuration uses environment variables that can be overridden through the docker-compose.yml file or by setting them directly in your environment.
- Database initialization scripts are located in the `mysql-init` and `postgres-init` directories.
