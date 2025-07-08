package com.ctuconnect.dto;

import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class FriendsDTO {
    private String userId;
    private List<String> friendIds;
    private List<UserDTO> friends;
    private List<UserDTO> mutualFriends;
    private List<UserDTO> friendSuggestions;
    private int mutualFriendsCount;

    // Constructor chỉ với danh sách friends
    public FriendsDTO(List<UserDTO> friends) {
        this.friends = friends;
    }

    // Constructor chỉ với danh sách mutual friends
    public static FriendsDTO ofMutualFriends(List<UserDTO> mutualFriends) {
        FriendsDTO dto = new FriendsDTO();
        dto.setMutualFriends(mutualFriends);
        return dto;
    }

    // Constructor chỉ với danh sách friend suggestions
    public static FriendsDTO ofSuggestions(List<UserDTO> suggestions) {
        FriendsDTO dto = new FriendsDTO();
        dto.setFriendSuggestions(suggestions);
        return dto;
    }
}