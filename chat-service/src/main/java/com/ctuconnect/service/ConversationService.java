package com.ctuconnect.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import com.ctuconnect.dto.request.CreateConversationRequest;
import com.ctuconnect.dto.request.UpdateConversationRequest;
import com.ctuconnect.dto.response.ConversationResponse;
import com.ctuconnect.dto.response.ParticipantInfo;
import com.ctuconnect.exception.ChatException;
import com.ctuconnect.model.Conversation;
import com.ctuconnect.repository.ConversationRepository;
import com.ctuconnect.repository.MessageRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class ConversationService {

    private final ConversationRepository conversationRepository;
    private final MessageRepository messageRepository;
    private final UserPresenceService userPresenceService;
    private final UserService userService;

    public ConversationResponse createConversation(CreateConversationRequest request, String createdBy) {
        log.info("Creating conversation with type: {} by user: {}", request.getType(), createdBy);

        // Validate participants
        if (request.getParticipantIds().isEmpty()) {
            throw new ChatException("Danh sách thành viên không được trống");
        }

        // Kiểm tra nếu là chat trực tiếp giữa 2 người đã tồn tại
        if (request.getType() == Conversation.ConversationType.DIRECT &&
            request.getParticipantIds().size() == 1) {

            String otherUserId = request.getParticipantIds().get(0);
            Optional<Conversation> existingConversation =
                conversationRepository.findDirectConversationBetweenUsers(createdBy, otherUserId);

            if (existingConversation.isPresent()) {
                return convertToResponse(existingConversation.get());
            }
        }

        // Tạo conversation mới
        Conversation conversation = new Conversation();
        conversation.setName(request.getName());
        conversation.setType(request.getType());

        // Thêm creator vào danh sách participants
        List<String> participants = request.getParticipantIds();
        if (!participants.contains(createdBy)) {
            participants.add(createdBy);
        }
        conversation.setParticipantIds(participants);

        conversation.setCreatedBy(createdBy);
        conversation.setCreatedAt(LocalDateTime.now());
        conversation.setUpdatedAt(LocalDateTime.now());

        // Metadata cho group chat
        if (request.getType() == Conversation.ConversationType.GROUP) {
            Conversation.ConversationMetadata metadata = new Conversation.ConversationMetadata();
            metadata.setDescription(request.getDescription());
            metadata.setAvatarUrl(request.getAvatarUrl());
            metadata.getAdminIds().add(createdBy); // Creator là admin
            conversation.setMetadata(metadata);
        }

        Conversation saved = conversationRepository.save(conversation);
        log.info("Created conversation: {} with {} participants", saved.getId(), participants.size());

        return convertToResponse(saved);
    }

    public Page<ConversationResponse> getUserConversations(String userId, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("lastMessageAt").descending());
        Page<Conversation> conversations = conversationRepository.findByParticipantIdsContaining(userId, pageable);

        return conversations.map(this::convertToResponse);
    }

    public ConversationResponse getConversationById(String conversationId, String userId) {
        Conversation conversation = conversationRepository.findById(conversationId)
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        // Kiểm tra quyền truy cập
        if (!conversation.getParticipantIds().contains(userId)) {
            throw new ChatException("Bạn không có quyền truy cập cuộc trò chuyện này");
        }

        return convertToResponse(conversation);
    }

    public void addParticipant(String conversationId, String participantId, String addedBy) {
        Conversation conversation = conversationRepository.findById(conversationId)
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        if (!canAddMembers(conversation, addedBy)) {
            throw new ChatException("Bạn không có quyền thêm thành viên");
        }

        if (!conversation.getParticipantIds().contains(participantId)) {
            conversation.getParticipantIds().add(participantId);
            conversation.setUpdatedAt(LocalDateTime.now());
            conversationRepository.save(conversation);

            log.info("Added participant {} to conversation {} by {}", participantId, conversationId, addedBy);
        }
    }

    public void removeParticipant(String conversationId, String participantId, String removedBy) {
        Conversation conversation = conversationRepository.findById(conversationId)
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        // Chỉ admin hoặc chính user đó mới có thể remove
        if (!canRemoveMembers(conversation, removedBy) && !participantId.equals(removedBy)) {
            throw new ChatException("Bạn không có quyền xóa thành viên");
        }

        conversation.getParticipantIds().remove(participantId);
        conversation.setUpdatedAt(LocalDateTime.now());
        conversationRepository.save(conversation);

        log.info("Removed participant {} from conversation {} by {}", participantId, conversationId, removedBy);
    }

    public ConversationResponse updateConversation(String conversationId, UpdateConversationRequest request, String userId) {
        Conversation conversation = conversationRepository.findById(conversationId)
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        // Kiểm tra quyền chỉnh sửa
        if (!canEditConversation(conversation, userId)) {
            throw new ChatException("Bạn không có quyền chỉnh sửa cuộc trò chuyện này");
        }

        // Cập nhật thông tin
        if (request.getName() != null) {
            conversation.setName(request.getName());
        }

        if (conversation.getMetadata() != null) {
            if (request.getDescription() != null) {
                conversation.getMetadata().setDescription(request.getDescription());
            }
            if (request.getAvatarUrl() != null) {
                conversation.getMetadata().setAvatarUrl(request.getAvatarUrl());
            }
        }

        conversation.setUpdatedAt(LocalDateTime.now());
        Conversation updated = conversationRepository.save(conversation);

        log.info("Updated conversation {} by user {}", conversationId, userId);
        return convertToResponse(updated);
    }

    public List<ConversationResponse> searchConversations(String userId, String query) {
        List<Conversation> conversations = conversationRepository
            .findByParticipantIdsContainingAndNameContainingIgnoreCase(userId, query);

        return conversations.stream()
            .map(this::convertToResponse)
            .collect(Collectors.toList());
    }

    private ConversationResponse convertToResponse(Conversation conversation) {
        ConversationResponse response = new ConversationResponse();
        response.setId(conversation.getId());
        response.setName(conversation.getName());
        response.setType(conversation.getType());
        response.setCreatedAt(conversation.getCreatedAt());
        response.setUpdatedAt(conversation.getUpdatedAt());
        response.setLastMessageAt(conversation.getLastMessageAt());

        // Lấy thông tin participants
        List<ParticipantInfo> participants = conversation.getParticipantIds().stream()
            .map(this::getParticipantInfo)
            .collect(Collectors.toList());
        response.setParticipants(participants);

        // Avatar cho group chat
        if (conversation.getMetadata() != null) {
            response.setAvatarUrl(conversation.getMetadata().getAvatarUrl());
        }

        // Đếm unread messages (placeholder)
        response.setUnreadCount(0);

        return response;
    }

    private ParticipantInfo getParticipantInfo(String userId) {
        ParticipantInfo info = new ParticipantInfo();
        info.setUserId(userId);
        // TODO: Implement thực tế sau khi tích hợp với UserService
        info.setUserName("User " + userId);
        info.setUserAvatar("");
        return info;
    }

    private boolean canAddMembers(Conversation conversation, String userId) {
        if (conversation.getSettings() != null &&
            !conversation.getSettings().isAllowMembersToAddOthers()) {
            return conversation.getMetadata() != null &&
                   conversation.getMetadata().getAdminIds().contains(userId);
        }
        return conversation.getParticipantIds().contains(userId);
    }

    private boolean canRemoveMembers(Conversation conversation, String userId) {
        return conversation.getMetadata() != null &&
               conversation.getMetadata().getAdminIds().contains(userId);
    }

    private boolean canEditConversation(Conversation conversation, String userId) {
        // Direct conversation không thể edit
        if (conversation.getType() == Conversation.ConversationType.DIRECT) {
            return false;
        }

        // Group conversation: chỉ admin mới được edit
        return conversation.getMetadata() != null &&
               conversation.getMetadata().getAdminIds().contains(userId);
    }

    public void deleteConversation(String conversationId, String userId) {
        Conversation conversation = conversationRepository.findById(conversationId)
            .orElseThrow(() -> new ChatException("Không tìm thấy cuộc trò chuyện"));

        // Chỉ admin hoặc creator mới có thể xóa
        if (!conversation.getCreatedBy().equals(userId) &&
            (conversation.getMetadata() == null || !conversation.getMetadata().getAdminIds().contains(userId))) {
            throw new ChatException("Bạn không có quyền xóa cuộc trò chuyện này");
        }

        // Xóa tất cả messages trong conversation
        messageRepository.deleteByConversationId(conversationId);

        // Xóa conversation
        conversationRepository.delete(conversation);
        log.info("Deleted conversation {} by user {}", conversationId, userId);
    }
}
