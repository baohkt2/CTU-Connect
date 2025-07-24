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

@Service
@RequiredArgsConstructor
@Slf4j
public class EmailService {

    private final JavaMailSender mailSender;
    private final TemplateEngine templateEngine;

    @Value("${spring.mail.username}")
    private String fromEmail;

    @Value("${app.frontend-url:http://localhost:3000}")

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
            helper.setSubject("CTU Connect - Xác thực email của bạn");

            // Create the verification link
            String verificationLink = frontendUrl + "/verify-email?token=" + token;
            String username = toEmail.split("@")[0];

            // Create HTML content using String.format instead of text block with .formatted()
            String htmlContent = String.format("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>CTU Connect - Xác thực email</title>
                        <style>
                            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                            .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
                            .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
                            .button { display: inline-block; background: #4266f5; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
                            .footer { text-align: center; margin-top: 30px; font-size: 12px; color: #666; }
                            .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>🎓 CTU Connect</h1>
                                <p>Chào mừng bạn đến với mạng xã hội sinh viên CTU!</p>
                            </div>
                            <div class="content">
                                <h2>Xin chào %s!</h2>
                                <p>Cảm ơn bạn đã đăng ký tài khoản CTU Connect. Để hoàn tất quá trình đăng ký, vui lòng xác thực địa chỉ email của bạn.</p>
                    
                                <div style="text-align: center; margin: 30px 0;">
                                    <a href="%s" class="button">✅ Xác thực email ngay</a>
                                </div>
                    
                                <div class="warning">
                                    <strong>⚠️ Lưu ý quan trọng:</strong>
                                    <ul>
                                        <li>Link xác thực có hiệu lực trong <strong>24 giờ</strong></li>
                                        <li>Chỉ nhấp vào link nếu bạn đã đăng ký tài khoản CTU Connect</li>
                                        <li>Nếu không thể nhấp vào nút, hãy sao chép link sau vào trình duyệt:</li>
                                    </ul>
                                    <p style="word-break: break-all; background: #e9ecef; padding: 10px; border-radius: 3px;">
                                        <code>%s</code>
                                    </p>
                                </div>
                    
                                <p>Sau khi xác thực thành công, bạn có thể:</p>
                                <ul>
                                    <li>🔑 Đăng nhập vào tài khoản</li>
                                    <li>📝 Tạo profile và kết nối với bạn bè</li>
                                    <li>💬 Tham gia vào các cuộc trò chuyện</li>
                                    <li>📚 Chia sẻ kinh nghiệm học tập</li>
                                </ul>
                            </div>
                            <div class="footer">
                                <p>Email này được gửi tự động từ hệ thống CTU Connect.<br>
                                   Nếu bạn không đăng ký tài khoản, vui lòng bỏ qua email này.</p>
                                <p>© 2025 CTU Connect - Mạng xã hội sinh viên Đại học Cần Thơ</p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """, username, verificationLink, verificationLink);

            helper.setText(htmlContent, true);
            mailSender.send(message);
            log.info("Verification email sent to: {}", toEmail);
        } catch (MessagingException e) {
            log.error("Failed to send verification email to {}: {}", toEmail, e.getMessage());
            throw new RuntimeException("Failed to send verification email", e);
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
            helper.setSubject("CTU Connect - Yêu cầu đặt lại mật khẩu");

            String resetLink = frontendUrl + "/reset-password?token=" + token;
            String username = toEmail.split("@")[0];

            String htmlContent = """
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>CTU Connect - Đặt lại mật khẩu</title>
                            <style>
                                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                                .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
                                .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
                                .button { display: inline-block; background:#4266f5; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
                                .footer { text-align: center; margin-top: 30px; font-size: 12px; color: #666; }
                                .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                <div class="header">
                                    <h1>🔐 CTU Connect</h1>
                                    <p>Khôi phục quyền truy cập vào tài khoản của bạn</p>
                                </div>
                                <div class="content">
                                    <h2>Xin chào %s,</h2>
                                    <p>Chúng tôi nhận được yêu cầu đặt lại mật khẩu cho tài khoản của bạn.</p>
                                    <div style="text-align: center; margin: 30px 0;">
                                        <a href="%s" class="button">🔑 Đặt lại mật khẩu</a>
                                    </div>
                                    <div class="warning">
                                        <strong>⚠️ Lưu ý quan trọng:</strong>
                                        <ul>
                                            <li>Link này chỉ có hiệu lực trong <strong>1 giờ</strong></li>
                                            <li>Nếu bạn không yêu cầu đặt lại mật khẩu, hãy bỏ qua email này</li>
                                            <li>Nếu nút không hoạt động, sao chép link bên dưới và dán vào trình duyệt:</li>
                                        </ul>
                                        <p style="word-break: break-all; background: #e9ecef; padding: 10px; border-radius: 3px;">
                                            <code>%s</code>
                                        </p>
                                    </div>
                                    <p>Sau khi đặt lại mật khẩu, bạn có thể tiếp tục sử dụng tài khoản như bình thường.</p>
                                </div>
                                <div class="footer">
                                    <p>Email này được gửi tự động từ hệ thống CTU Connect.<br>
                                       Nếu bạn không thực hiện yêu cầu, vui lòng bỏ qua email này.</p>
                                    <p>© 2025 CTU Connect - Mạng xã hội sinh viên Đại học Cần Thơ</p>
                                </div>
                            </div>
                        </body>
                        </html>
                    """.formatted(username, resetLink, resetLink);


            helper.setText(htmlContent, true);
            mailSender.send(message);
            log.info("Password reset email sent to: {}", toEmail);
        } catch (MessagingException e) {
            log.error("Failed to send password reset email to {}: {}", toEmail, e.getMessage());
        }
    }

}
