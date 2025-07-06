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
}