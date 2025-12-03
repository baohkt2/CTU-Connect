package com.ctuconnect.service;

import com.ctuconnect.dto.response.MessageAttachmentResponse;
import com.ctuconnect.dto.response.MessageReactionResponse;
import com.ctuconnect.model.Conversation;
import com.ctuconnect.model.Message;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import com.ctuconnect.dto.request.AddReactionRequest;
import com.ctuconnect.dto.request.SendMessageRequest;
import com.ctuconnect.dto.response.MessageResponse;
import com.ctuconnect.dto.response.ChatPageResponse;
import com.ctuconnect.exception.ChatException;
import com.ctuconnect.model.*;
import com.ctuconnect.repository.ConversationRepository;
import com.ctuconnect.repository.MessageRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class MessageService {

    private final MessageRepository messageRepository;
    private final ConversationRepository conversationRepository;
    private final WebSocketService webSocketService;
    private final NotificationService notificationService;

    public MessageResponse sendMessage(SendMessageRequest request, String senderId) {
        log.info("Sending message from user: {} to conversation: {}", senderId, request.getConversationId());

        // Kiểm tra conversation tồn tại và user có quyền gửi tin nhắn
        Conversation conversation = conversationRepository.findById(request.getConversationId())
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        if (!conversation.getParticipantIds().contains(senderId)) {
            throw new ChatException("Bạn không có quyền gửi tin nhắn trong cuộc trò chuyện này");
        }

        // Tạo message mới
        Message message = new Message();
        message.setConversationId(request.getConversationId());
        message.setSenderId(senderId);
        message.setType(Message.MessageType.TEXT);
        message.setContent(request.getContent());
        message.setReplyToMessageId(request.getReplyToMessageId());
        message.setStatus(Message.MessageStatus.SENT);
        message.setCreatedAt(LocalDateTime.now());
        message.setUpdatedAt(LocalDateTime.now());

        // Lấy thông tin sender (cache)
        // TODO: Tích hợp với UserService
        message.setSenderName("User " + senderId);
        message.setSenderAvatar("");

        // Lưu message
        Message savedMessage = messageRepository.save(message);

        // Cập nhật last message của conversation
        conversation.setLastMessageId(savedMessage.getId());
        conversation.setLastMessageAt(savedMessage.getCreatedAt());
        conversation.setUpdatedAt(LocalDateTime.now());
        conversationRepository.save(conversation);

        // Chuyển đổi thành response
        MessageResponse response = convertToResponse(savedMessage);

        // Gửi real-time notification qua WebSocket
        webSocketService.sendMessageToConversation(request.getConversationId(), response);

        // Gửi push notification
        notificationService.sendMessageNotification(conversation, savedMessage);

        log.info("Message sent successfully: {}", savedMessage.getId());
        return response;
    }

    public ChatPageResponse<MessageResponse> getMessages(String conversationId, String userId, int page, int size) {
        // Kiểm tra quyền truy cập
        Conversation conversation = conversationRepository.findById(conversationId)
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        if (!conversation.getParticipantIds().contains(userId)) {
            throw new ChatException("Bạn không có quyền truy cập cuộc trò chuyện này");
        }

        // Lấy messages với pagination (sắp xếp theo thời gian giảm dần)
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        Page<Message> messages = messageRepository.findByConversationIdAndIsDeletedFalseOrderByCreatedAtDesc(
            conversationId, pageable);

        // Chuyển đổi thành response
        List<MessageResponse> messageResponses = messages.getContent().stream()
            .map(this::convertToResponse)
            .collect(Collectors.toList());

        // Đánh dấu messages là đã đọc
        markMessagesAsRead(conversationId, userId);

        return new ChatPageResponse<>(
            messageResponses,
            messages.getNumber(),
            messages.getSize(),
            messages.getTotalElements(),
            messages.getTotalPages(),
            messages.hasNext(),
            messages.hasPrevious()
        );
    }

    public MessageResponse editMessage(String messageId, String newContent, String userId) {
        Message message = messageRepository.findById(messageId)
            .orElseThrow(() -> new ChatException("Không tìm thấy tin nhắn"));

        // Chỉ người gửi mới có thể sửa tin nhắn
        if (!message.getSenderId().equals(userId)) {
            throw new ChatException("Bạn không có quyền sửa tin nhắn này");
        }

        // Không thể sửa tin nhắn quá cũ (15 phút)
        if (message.getCreatedAt().isBefore(LocalDateTime.now().minusMinutes(15))) {
            throw new ChatException("Không thể sửa tin nhắn sau 15 phút");
        }

        message.setContent(newContent);
        message.setEdited(true);
        message.setEditedAt(LocalDateTime.now());
        message.setUpdatedAt(LocalDateTime.now());

        Message updated = messageRepository.save(message);
        MessageResponse response = convertToResponse(updated);

        // Gửi cập nhật real-time
        webSocketService.sendMessageUpdateToConversation(message.getConversationId(), response);

        return response;
    }

    public void deleteMessage(String messageId, String userId) {
        Message message = messageRepository.findById(messageId)
            .orElseThrow(() -> new ChatException("Không tìm thấy tin nhắn"));

        // Chỉ người gửi mới có thể xóa tin nhắn
        if (!message.getSenderId().equals(userId)) {
            throw new ChatException("Bạn không có quyền xóa tin nhắn này");
        }

        message.setDeleted(true);
        message.setContent("Tin nhắn đã được xóa");
        message.setUpdatedAt(LocalDateTime.now());

        messageRepository.save(message);

        // Gửi cập nhật real-time
        webSocketService.sendMessageDeleteToConversation(message.getConversationId(), messageId);

        log.info("Message {} deleted by user {}", messageId, userId);
    }

    public MessageResponse addReaction(AddReactionRequest request, String userId) {
        Message message = messageRepository.findById(request.getMessageId())
            .orElseThrow(() -> new ChatException("Không tìm thấy tin nhắn"));

        // Kiểm tra user có quyền react không
        Conversation conversation = conversationRepository.findById(message.getConversationId())
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        if (!conversation.getParticipantIds().contains(userId)) {
            throw new ChatException("Bạn không có quyền react tin nhắn này");
        }

        // Kiểm tra xem user đã react chưa
        Optional<Message.MessageReaction> existingReaction = message.getReactions().stream()
            .filter(r -> r.getUserId().equals(userId))
            .findFirst();

        if (existingReaction.isPresent()) {
            // Update existing reaction
            existingReaction.get().setEmoji(request.getEmoji());
            existingReaction.get().setCreatedAt(LocalDateTime.now());
        } else {
            // Add new reaction
            Message.MessageReaction reaction = new Message.MessageReaction();
            reaction.setUserId(userId);
            reaction.setUserName("User " + userId); // TODO: Get from UserService
            reaction.setEmoji(request.getEmoji());
            reaction.setCreatedAt(LocalDateTime.now());
            message.getReactions().add(reaction);
        }

        message.setUpdatedAt(LocalDateTime.now());
        Message updated = messageRepository.save(message);

        MessageResponse response = convertToResponse(updated);

        // Gửi cập nhật real-time
        webSocketService.sendReactionUpdateToConversation(message.getConversationId(), response);

        return response;
    }

    public void removeReaction(String messageId, String userId) {
        Message message = messageRepository.findById(messageId)
            .orElseThrow(() -> new ChatException("Không tìm thấy tin nhắn"));

        message.getReactions().removeIf(r -> r.getUserId().equals(userId));
        message.setUpdatedAt(LocalDateTime.now());

        Message updated = messageRepository.save(message);
        MessageResponse response = convertToResponse(updated);

        // Gửi cập nhật real-time
        webSocketService.sendReactionUpdateToConversation(message.getConversationId(), response);
    }

    public List<MessageResponse> searchMessages(String conversationId, String query, String userId) {
        // Kiểm tra quyền truy cập
        Conversation conversation = conversationRepository.findById(conversationId)
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        if (!conversation.getParticipantIds().contains(userId)) {
            throw new ChatException("Bạn không có quyền tìm kiếm trong cuộc trò chuyện này");
        }

        List<Message> messages = messageRepository.searchMessagesInConversation(conversationId, query);

        return messages.stream()
            .map(this::convertToResponse)
            .collect(Collectors.toList());
    }

    public long getUnreadCount(String conversationId, String userId) {
        return messageRepository.countUnreadMessages(conversationId, userId);
    }

    public void markMessagesAsRead(String conversationId, String userId) {
        // Lấy tất cả messages chưa đọc trong conversation
        List<Message> unreadMessages = messageRepository.findByConversationIdAndIsDeletedFalseOrderByCreatedAtDesc(
            conversationId, PageRequest.of(0, 100)).getContent().stream()
            .filter(m -> !m.getReadByUserIds().contains(userId) && !m.getSenderId().equals(userId))
            .collect(Collectors.toList());

        // Đánh dấu là đã đọc
        for (Message message : unreadMessages) {
            if (!message.getReadByUserIds().contains(userId)) {
                message.getReadByUserIds().add(userId);
                message.setStatus(Message.MessageStatus.READ);
                message.setUpdatedAt(LocalDateTime.now());
            }
        }

        if (!unreadMessages.isEmpty()) {
            messageRepository.saveAll(unreadMessages);

            // Gửi read receipt real-time
            webSocketService.sendReadReceiptToConversation(conversationId, userId);
        }
    }

    private MessageResponse convertToResponse(Message message) {
        MessageResponse response = new MessageResponse();
        response.setId(message.getId());
        response.setConversationId(message.getConversationId());
        response.setSenderId(message.getSenderId());
        response.setSenderName(message.getSenderName());
        response.setSenderAvatar(message.getSenderAvatar());
        response.setType(message.getType());
        response.setContent(message.getContent());
        response.setReplyToMessageId(message.getReplyToMessageId());
        response.setStatus(message.getStatus());
        response.setReadByUserIds(message.getReadByUserIds());
        response.setCreatedAt(message.getCreatedAt());
        response.setUpdatedAt(message.getUpdatedAt());
        response.setEditedAt(message.getEditedAt());
        response.setEdited(message.isEdited());
        response.setDeleted(message.isDeleted());

        // Convert reactions
        if (message.getReactions() != null) {
            response.setReactions(
                message.getReactions().stream()
                    .map(r -> new MessageReactionResponse(r.getUserId(), r.getUserName(), r.getEmoji(), r.getCreatedAt()))
                    .collect(Collectors.toList())
            );
        }

        // Convert attachment if exists
        if (message.getAttachment() != null) {
            MessageAttachmentResponse attachment = new MessageAttachmentResponse();
            attachment.setFileName(message.getAttachment().getFileName());
            attachment.setFileUrl(message.getAttachment().getFileUrl());
            attachment.setFileType(message.getAttachment().getFileType());
            attachment.setFileSize(message.getAttachment().getFileSize());
            attachment.setThumbnailUrl(message.getAttachment().getThumbnailUrl());
            response.setAttachment(attachment);
        }

        // Load reply message if exists
        if (message.getReplyToMessageId() != null) {
            messageRepository.findById(message.getReplyToMessageId())
                .ifPresent(replyMessage -> response.setReplyToMessage(convertToResponse(replyMessage)));
        }

        return response;
    }
}
