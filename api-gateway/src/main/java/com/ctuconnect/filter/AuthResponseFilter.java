package com.ctuconnect.filter;

import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;

@Component
public class AuthResponseFilter implements GlobalFilter, Ordered {

    // Đây là ví dụ về cách bạn có thể nhận token từ response của auth-service.
    // Trong thực tế, bạn có thể cần đọc từ body JSON hoặc một header tùy chỉnh.
    // Giả sử auth-service trả về token trong một header tên là "X-Auth-Token"
    // hoặc "X-Refresh-Token". Nếu trả về trong body, bạn sẽ cần một cách phức tạp hơn
    // để đọc body response.

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        // Tiếp tục chuỗi filter để request được chuyển đến downstream service
        return chain.filter(exchange).then(Mono.fromRunnable(() -> {
            // Logic này sẽ chạy sau khi downstream service (auth-service) đã trả về response
            ServerHttpResponse response = exchange.getResponse();
            ServerHttpRequest request = exchange.getRequest(); // Để kiểm tra path

            // Chỉ áp dụng logic này cho các request đăng nhập/refresh token
            // Ví dụ: Nếu request đến path /api/auth/login hoặc /api/auth/refresh-token
            if (request.getURI().getPath().equals("/api/auth/login") ||
                    request.getURI().getPath().equals("/api/auth/refresh-token")) {

                // Kiểm tra xem response có thành công không
                if (response.getStatusCode() == HttpStatus.OK) {
                    HttpHeaders headers = response.getHeaders();

                    // Giả sử auth-service trả về access token và refresh token trong headers
                    // Hoặc bạn có thể đọc từ body nếu auth-service trả về JSON body
                    // (Đọc body response trong Gateway filter phức tạp hơn, cần DataBufferUtils)
                    List<String> accessTokenHeaders = headers.get("X-Access-Token");
                    List<String> refreshTokenHeaders = headers.get("X-Refresh-Token");

                    String accessToken = (accessTokenHeaders != null && !accessTokenHeaders.isEmpty()) ? accessTokenHeaders.get(0) : null;
                    String refreshToken = (refreshTokenHeaders != null && !refreshTokenHeaders.isEmpty()) ? refreshTokenHeaders.get(0) : null;

                    if (accessToken != null) {
                        // Thêm Access Token vào Cookie
                        ResponseCookie accessTokenCookie = ResponseCookie.from("access_token", accessToken)
                                .httpOnly(true) // Không cho JavaScript truy cập
                                .secure(true)   // Chỉ gửi qua HTTPS
                                .path("/")      // Có sẵn trên toàn bộ domain
                                .maxAge(Duration.ofHours(24)) // Thời gian sống của cookie (phù hợp với JWT expiration)
                                .build();
                        response.addCookie(accessTokenCookie);
                        System.out.println("Access Token added to cookie for user: " + request.getURI().getPath());
                    }

                    if (refreshToken != null) {
                        // Thêm Refresh Token vào Cookie
                        ResponseCookie refreshTokenCookie = ResponseCookie.from("refresh_token", refreshToken)
                                .httpOnly(true)
                                .secure(true)
                                .path("/api/auth/refresh-token") // Chỉ gửi cho endpoint refresh-token
                                .maxAge(Duration.ofDays(7)) // Thời gian sống của refresh token
                                .build();
                        response.addCookie(refreshTokenCookie);
                        System.out.println("Refresh Token added to cookie for user: " + request.getURI().getPath());
                    }

                    // Xóa các header token gốc từ response của auth-service
                    // để client chỉ nhận token qua cookie
                    headers.remove("X-Access-Token");
                    headers.remove("X-Refresh-Token");
                }
            }
        }));
    }

    @Override
    public int getOrder() {
        // Đảm bảo filter này chạy sau các filter khác (ví dụ: sau khi request được định tuyến)
        // Giá trị càng cao, độ ưu tiên càng thấp (chạy sau)
        return Ordered.LOWEST_PRECEDENCE;
    }
}
