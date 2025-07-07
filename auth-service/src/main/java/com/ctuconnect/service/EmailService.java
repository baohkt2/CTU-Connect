package com.ctuconnect.service;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.thymeleaf.TemplateEngine;
import org.thymeleaf.context.Context;

@Service
@RequiredArgsConstructor
@Slf4j
public class EmailService {

    private final JavaMailSender mailSender;
    private final TemplateEngine templateEngine;

    @Value("${spring.mail.username}")
    private String fromEmail;

    @Value("${app.frontend-url:http://localhost:8090/api/auth}")

    // Default to localhost for development; change in production
    private String frontendUrl;

    /**
     * Send verification email asynchronously
     */
    @Async
    public void sendVerificationEmail(String toEmail, String token) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");

            helper.setFrom(fromEmail);
            helper.setTo(toEmail);
            helper.setSubject("CTU Connect - Email Verification");

            // Create the verification link
            String verificationLink = frontendUrl + "/verify-email?token=" + token;

            // Create context for Thymeleaf template
            Context context = new Context();
            context.setVariable("verificationLink", verificationLink);
            context.setVariable("username", toEmail.split("@")[0]);

            // Use Thymeleaf template to generate HTML content
            // For simplicity, we'll use inline HTML instead of a template file
            String htmlContent = "<html><body>"
                + "<h2>Welcome to CTU Connect!</h2>"
                + "<p>Thank you for registering. Please verify your email by clicking the link below:</p>"
                + "<p><a href='" + verificationLink + "'>Verify Email Address</a></p>"
                + "<p>This link will expire in 24 hours.</p>"
                + "<p>If you did not create an account, please ignore this email.</p>"
                + "</body></html>";

            helper.setText(htmlContent, true);

            mailSender.send(message);
            log.info("Verification email sent to: {}", toEmail);
        } catch (MessagingException e) {
            log.error("Failed to send verification email to {}: {}", toEmail, e.getMessage());
        }
    }

    /**
     * Send password reset email asynchronously
     */
    @Async
    public void sendPasswordResetEmail(String toEmail, String token) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");

            helper.setFrom(fromEmail);
            helper.setTo(toEmail);
            helper.setSubject("CTU Connect - Password Reset");

            // Create the password reset link
            String resetLink = frontendUrl + "/reset-password?token=" + token;

            // Create context for Thymeleaf template
            Context context = new Context();
            context.setVariable("resetLink", resetLink);
            context.setVariable("username", toEmail.split("@")[0]);

            // Use Thymeleaf template to generate HTML content
            // For simplicity, we'll use inline HTML instead of a template file
            String htmlContent = "<html><body>"
                + "<h2>Password Reset Request</h2>"
                + "<p>We received a request to reset your password. Click the link below to reset it:</p>"
                + "<p><a href='" + resetLink + "'>Reset Password</a></p>"
                + "<p>This link will expire in 1 hour.</p>"
                + "<p>If you did not request a password reset, please ignore this email.</p>"
                + "</body></html>";

            helper.setText(htmlContent, true);

            mailSender.send(message);
            log.info("Password reset email sent to: {}", toEmail);
        } catch (MessagingException e) {
            log.error("Failed to send password reset email to {}: {}", toEmail, e.getMessage());
        }
    }
}
