package com.ctuconnect.dto.request;

import com.ctuconnect.model.Conversation;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CreateConversationRequest {
    @Size(max = 100, message = "Tên nhóm không được vượt quá 100 ký tự")
    private String name;

    @NotEmpty(message = "Danh sách thành viên không được trống")
    private List<String> participantIds;

    private Conversation.ConversationType type = Conversation.ConversationType.DIRECT;

    private String description;

    private String avatarUrl;
}
