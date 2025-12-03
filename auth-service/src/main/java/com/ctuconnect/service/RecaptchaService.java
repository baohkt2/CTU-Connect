package com.ctuconnect.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@Service
@Slf4j
public class RecaptchaService {

    @Value("${recaptcha.secret.key}")
    private String secretKey;

    @Value("${recaptcha.verify.url}")
    private String verifyUrl;

    private final WebClient webClient;

    public RecaptchaService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.build();
    }

    public boolean verifyRecaptcha(String token) {
        try {
            RecaptchaResponse response = webClient.post()
                    .uri(verifyUrl)
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .bodyValue("secret=" + secretKey + "&response=" + token)
                    .retrieve()
                    .bodyToMono(RecaptchaResponse.class)
                    .block();

            if (response != null) {
                boolean isValid = response.isSuccess() && response.getScore() >= 0.5;
                log.info("reCAPTCHA verification result: success={}, score={}, action={}",
                        response.isSuccess(), response.getScore(), response.getAction());
                return isValid;
            }

            log.warn("reCAPTCHA verification failed: null response");
            return false;
        } catch (Exception e) {
            log.error("Error verifying reCAPTCHA token", e);
            return false;
        }
    }

    public static class RecaptchaResponse {
        private boolean success;
        private double score;
        private String action;
        private String challenge_ts;
        private String hostname;
        private String[] error_codes;

        // Getters and setters
        public boolean isSuccess() {
            return success;
        }

        public void setSuccess(boolean success) {
            this.success = success;
        }

        public double getScore() {
            return score;
        }

        public void setScore(double score) {
            this.score = score;
        }

        public String getAction() {
            return action;
        }

        public void setAction(String action) {
            this.action = action;
        }

        public String getChallenge_ts() {
            return challenge_ts;
        }

        public void setChallenge_ts(String challenge_ts) {
            this.challenge_ts = challenge_ts;
        }

        public String getHostname() {
            return hostname;
        }

        public void setHostname(String hostname) {
            this.hostname = hostname;
        }

        public String[] getError_codes() {
            return error_codes;
        }

        public void setError_codes(String[] error_codes) {
            this.error_codes = error_codes;
        }
    }
}
