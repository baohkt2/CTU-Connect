package com.ctuconnect.repository;

import com.ctuconnect.model.Message;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface MessageRepository extends MongoRepository<Message, String> {
    
    // Lấy messages trong conversation với pagination
    Page<Message> findByConversationIdAndIsDeletedFalseOrderByCreatedAtDesc(String conversationId, Pageable pageable);
    
    // Lấy tin nhắn mới nhất của conversation
    Optional<Message> findFirstByConversationIdAndIsDeletedFalseOrderByCreatedAtDesc(String conversationId);
    
    // Tìm kiếm messages theo nội dung
    @Query("{'conversationId': ?0, 'content': {$regex: ?1, $options: 'i'}, 'isDeleted': false}")
    List<Message> searchMessagesInConversation(String conversationId, String searchText);
    
    // Đếm messages chưa đọc trong conversation
    @Query(value = "{'conversationId': ?0, 'senderId': {$ne: ?1}, 'readByUserIds': {$nin: [?1]}, 'isDeleted': false}", count = true)
    long countUnreadMessages(String conversationId, String userId);
    
    // Lấy messages sau một thời điểm cụ thể (real-time sync)
    List<Message> findByConversationIdAndCreatedAtAfterAndIsDeletedFalseOrderByCreatedAtAsc(
        String conversationId, LocalDateTime after);
    
    // Lấy messages được gửi bởi user cụ thể
    List<Message> findBySenderIdAndConversationIdAndIsDeletedFalseOrderByCreatedAtDesc(
        String senderId, String conversationId);
    
    // Lấy messages có attachments
    @Query("{'conversationId': ?0, 'attachment': {$ne: null}, 'isDeleted': false}")
    List<Message> findMessagesWithAttachments(String conversationId);
    
    // Xóa tất cả messages trong conversation
    void deleteByConversationId(String conversationId);
}
