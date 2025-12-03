package com.ctuconnect.exception;

public class ChatException extends RuntimeException {
    public ChatException(String message) {
        super(message);
    }

    public ChatException(String message, Throwable cause) {
        super(message, cause);
    }
}

class ConversationNotFoundException extends ChatException {
    public ConversationNotFoundException(String conversationId) {
        super("Không tìm thấy cuộc trò chuyện với ID: " + conversationId);
    }
}

class MessageNotFoundException extends ChatException {
    public MessageNotFoundException(String messageId) {
        super("Không tìm thấy tin nhắn với ID: " + messageId);
    }
}

class UnauthorizedAccessException extends ChatException {
    public UnauthorizedAccessException(String message) {
        super("Không có quyền truy cập: " + message);
    }
}
