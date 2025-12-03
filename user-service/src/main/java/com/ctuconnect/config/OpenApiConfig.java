package com.ctuconnect.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("CTU Connect - User Service API")
                        .version("1.0.0")
                        .description("API documentation for CTU Connect User Service - " +
                                   "A comprehensive user management system for university social networking")
                        .contact(new Contact()
                                .name("CTU Connect Team")
                                .email("support@ctuconnect.edu.vn")
                                .url("https://ctuconnect.edu.vn"))
                        .license(new License()
                                .name("MIT License")
                                .url("https://opensource.org/licenses/MIT")))
                .servers(List.of(
                        new Server()
                                .url("http://localhost:8081/api/users")
                                .description("Development server"),
                        new Server()
                                .url("https://api.ctuconnect.edu.vn/users")
                                .description("Production server")));
    }
}
