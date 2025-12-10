package com.ctuconnect.repository;

import com.ctuconnect.model.Conversation;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface ConversationRepository extends MongoRepository<Conversation, String> {
    
    // Tìm conversations mà user tham gia
    @Query("{'participantIds': ?0}")
    Page<Conversation> findByParticipantIdsContaining(String userId, Pageable pageable);
    
    // Tìm conversation trực tiếp giữa 2 users - return List to handle duplicates
    @Query("{'type': 'DIRECT', 'participantIds': {$all: [?0, ?1], $size: 2}}")
    List<Conversation> findDirectConversationsBetweenUsers(String userId1, String userId2);
    
    // Tìm conversations theo tên (search)
    @Query("{'participantIds': ?0, 'name': {$regex: ?1, $options: 'i'}}")
    List<Conversation> findByParticipantIdsContainingAndNameContainingIgnoreCase(String userId, String name);
    
    // Tìm conversations được cập nhật gần đây
    @Query("{'participantIds': ?0, 'lastMessageAt': {$gte: ?1}}")
    List<Conversation> findRecentConversations(String userId, LocalDateTime since);
    
    // Đếm số conversations của user
    @Query(value = "{'participantIds': ?0}", count = true)
    long countByParticipantIdsContaining(String userId);
}
