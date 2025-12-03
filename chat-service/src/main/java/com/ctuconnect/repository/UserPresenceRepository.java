package com.ctuconnect.repository;

import com.ctuconnect.model.UserPresence;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface UserPresenceRepository extends MongoRepository<UserPresence, String> {
    
    // Tìm presence theo userId
    Optional<UserPresence> findByUserId(String userId);
    
    // Tìm presence theo sessionId
    Optional<UserPresence> findBySessionId(String sessionId);
    
    // Lấy danh sách users online
    List<UserPresence> findByStatus(UserPresence.PresenceStatus status);
    
    // Lấy presence của nhiều users
    List<UserPresence> findByUserIdIn(List<String> userIds);
    
    // Tìm users đang typing trong conversation
    @Query("{'currentActivity': {$regex: '^typing in ?0', $options: 'i'}}")
    List<UserPresence> findUsersTypingInConversation(String conversationId);
    
    // Lấy users online gần đây
    List<UserPresence> findByLastSeenAtAfterOrderByLastSeenAtDesc(LocalDateTime since);
    
    // Xóa presence của user khi disconnect
    void deleteByUserId(String userId);
    
    // Xóa presence cũ (cleanup job)
    void deleteByLastSeenAtBefore(LocalDateTime before);
}
