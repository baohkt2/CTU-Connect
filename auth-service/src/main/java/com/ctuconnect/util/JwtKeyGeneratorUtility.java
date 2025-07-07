package com.ctuconnect.util;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Encoders;
import javax.crypto.SecretKey;

public class JwtKeyGeneratorUtility {
    public static void main(String[] args) {
        // Tạo một khóa HS256 ngẫu nhiên, an toàn (256 bits)
        SecretKey key = Jwts.SIG.HS256.key().build();

        // Mã hóa khóa này sang định dạng Base64 URL-safe
        String base64Key = Encoders.BASE64.encode(key.getEncoded());

        System.out.println("--------------------------------------------------------------------");
        System.out.println("Generated HS256 Secret Key (Base64 Encoded):");
        System.out.println(base64Key);
        System.out.println("Key length in bits: " + (key.getEncoded().length * 8));
        System.out.println("--------------------------------------------------------------------");
        System.out.println("\nAdd this to your application.properties (or application.yml):");
        System.out.println("application.security.jwt.secret-key=" + base64Key);
        System.out.println("\nRemember to keep this key secret and do not share it publicly!");
    }
}
