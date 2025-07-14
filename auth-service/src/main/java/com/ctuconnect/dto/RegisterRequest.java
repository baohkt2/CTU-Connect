package com.ctuconnect.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class RegisterRequest {

    @NotBlank(message = "Email là bắt buộc")
    @Email(message = "Email phải có định dạng hợp lệ")
    @Pattern(regexp = "^[A-Za-z0-9._%+-]+@(student\\.)?ctu\\.edu\\.vn$",
            message = "Email phải có đuôi @ctu.edu.vn hoặc @student.ctu.edu.vn")
    private String email;

    @NotBlank(message = "Tên đăng nhập là bắt buộc")
    @Size(min = 3, max = 25, message = "Tên đăng nhập phải có từ 3-25 ký tự")
    @Pattern(regexp = "^[a-zA-Z][a-zA-Z0-9._]{2,24}$",
            message = "Tên đăng nhập phải bắt đầu bằng chữ cái và chỉ chứa chữ cái, số, dấu chấm và gạch dưới")
    private String username;

    @NotBlank(message = "Mật khẩu là bắt buộc")
    @Size(min = 8, max = 20, message = "Mật khẩu phải có từ 8-20 ký tự")
    @Pattern(regexp = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=!])(?!.*\\s).{8,20}$",
            message = "Mật khẩu phải chứa ít nhất: 1 chữ số, 1 chữ thường, 1 chữ hoa, 1 ký tự đặc biệt và không có khoảng trắng")
    private String password;

    // reCAPTCHA token
    @NotBlank(message = "reCAPTCHA token là bắt buộc")
    private String recaptchaToken;
}
