package com.ctuconnect.repository;

import com.ctuconnect.entity.NotificationEntity;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository  
public interface NotificationRepository extends MongoRepository<NotificationEntity, String> {
    
    List<NotificationEntity> findByRecipientIdOrderByCreatedAtDesc(String recipientId, Pageable pageable);
    
    List<NotificationEntity> findByRecipientIdAndIsReadFalse(String recipientId);
    
    long countByRecipientIdAndIsReadFalse(String recipientId);
    
    void deleteByCreatedAtBefore(LocalDateTime cutoffDate);
    
    @Query("{'recipientId': ?0, 'type': ?1}")
    List<NotificationEntity> findByRecipientIdAndType(String recipientId, NotificationEntity.NotificationType type);
    
    @Query("{'recipientId': ?0, 'actorId': ?1, 'entityId': ?2}")
    List<NotificationEntity> findDuplicateNotifications(String recipientId, String actorId, String entityId);
}
