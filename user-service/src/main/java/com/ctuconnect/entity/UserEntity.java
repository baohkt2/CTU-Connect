package com.ctuconnect.entity;

import lombok.Data;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

@Node("User")
@Data
public class UserEntity {
    @Id
    private String id;

    // Đã thêm: Trường email là cần thiết cho các phương thức của repository như findByEmail
    private String email;

    // Đã đổi tên: Tuân thủ quy ước đặt tên camelCase của Java (student_ID -> studentId)
    private String studentId;

    private String batch;
    private String fullName;
    private String role;
    private String college;
    private String faculty;
    private String major;
    private String gender;
    private String bio;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @Relationship(type = "FRIEND", direction = Relationship.Direction.OUTGOING)
    private Set<UserEntity> friends = new HashSet<>();
}