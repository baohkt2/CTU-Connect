// package com.ctuconnect.controller;

// import com.ctuconnect.dto.UserProfileDTO;
// import com.ctuconnect.dto.UserUpdateDTO;
// import com.ctuconnect.service.UserService;
// import com.fasterxml.jackson.databind.ObjectMapper;

// import org.junit.jupiter.api.BeforeEach;
// import org.junit.jupiter.api.Test;
// import org.springframework.beans.factory.annotation.Autowired;
// import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
// import org.springframework.boot.test.mock.mockito.MockBean;
// import org.springframework.http.MediaType;
// import org.springframework.test.web.servlet.MockMvc;

// import java.time.LocalDateTime;
// import java.util.Map;

// import static org.mockito.ArgumentMatchers.any;
// import static org.mockito.ArgumentMatchers.anyString;
// import static org.mockito.Mockito.when;
// import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
// import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

// @WebMvcTest(UserController.class)
// class UserControllerTest {

//     @Autowired
//     private MockMvc mockMvc;

//     @MockBean
//     private UserService userService;

//     @Autowired
//     private ObjectMapper objectMapper;

//     private UserProfileDTO testUserProfileDTO;

//     @BeforeEach
//     void setUp() {
//         testUserProfileDTO = UserProfileDTO.builder()
//                 .id("test-user-id")
//                 .email("test@example.com")
//                 .username("testuser")
//                 .fullName("Test User")
//                 .role("USER")
//                 .isActive(true)
//                 .createdAt(LocalDateTime.now())
//                 .updatedAt(LocalDateTime.now())
//                 .build();
//     }

//     @Test
//     void getUserProfile_ShouldReturnUserProfile() throws Exception {
//         // Arrange
//         when(userService.getUserProfile("test-user-id")).thenReturn(testUserProfileDTO);

//         // Act & Assert
//         mockMvc.perform(get("/users/test-user-id"))
//                 .andExpect(status().isOk())
//                 .andExpect(content().contentType(MediaType.APPLICATION_JSON))
//                 .andExpect(jsonPath("$.id").value("test-user-id"))
//                 .andExpect(jsonPath("$.email").value("test@example.com"))
//                 .andExpect(jsonPath("$.username").value("testuser"))
//                 .andExpect(jsonPath("$.fullName").value("Test User"));
//     }

//     @Test
//     void createUser_ShouldCreateUser() throws Exception {
//         // Arrange
//         Map<String, String> userRequest = Map.of(
//                 "authUserId", "auth-user-id",
//                 "email", "test@example.com",
//                 "username", "testuser",
//                 "role", "USER"
//         );

//         // Act & Assert
//         mockMvc.perform(post("/users")
//                         .contentType(MediaType.APPLICATION_JSON)
//                         .content(objectMapper.writeValueAsString(userRequest)))
//                 .andExpect(status().isCreated())
//                 .andExpect(jsonPath("$.message").value("User created successfully"))
//                 .andExpect(jsonPath("$.userId").value("auth-user-id"));
//     }

//     @Test
//     void updateUserProfile_ShouldUpdateProfile() throws Exception {
//         // Arrange
//         UserUpdateDTO updateDTO = UserUpdateDTO.builder()
//                 .fullName("Updated Name")
//                 .bio("Updated bio")
//                 .build();

//         when(userService.updateUserProfile(anyString(), any(UserUpdateDTO.class)))
//                 .thenReturn(testUserProfileDTO);

//         // Act & Assert
//         mockMvc.perform(put("/users/test-user-id")
//                         .contentType(MediaType.APPLICATION_JSON)
//                         .content(objectMapper.writeValueAsString(updateDTO)))
//                 .andExpect(status().isOk())
//                 .andExpect(content().contentType(MediaType.APPLICATION_JSON))
//                 .andExpect(jsonPath("$.id").value("test-user-id"));
//     }

//     @Test
//     void sendFriendRequest_ShouldSendRequest() throws Exception {
//         // Act & Assert
//         mockMvc.perform(post("/users/sender-id/friend-requests/receiver-id"))
//                 .andExpect(status().isOk())
//                 .andExpect(jsonPath("$.message").value("Friend request sent successfully"));
//     }

//     @Test
//     void userExists_ShouldReturnExistenceStatus() throws Exception {
//         // Arrange
//         when(userService.userExists("test-user-id")).thenReturn(true);

//         // Act & Assert
//         mockMvc.perform(get("/users/test-user-id/exists"))
//                 .andExpect(status().isOk())
//                 .andExpect(jsonPath("$.exists").value(true));
//     }
// }
