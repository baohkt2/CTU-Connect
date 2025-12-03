package com.ctuconnect.service;

import com.ctuconnect.dto.UserProfileDTO;
import com.ctuconnect.dto.UserUpdateDTO;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.exception.UserNotFoundException;
import com.ctuconnect.mapper.UserMapper;
import com.ctuconnect.repository.UserRepository;
import com.ctuconnect.repository.MajorRepository;
import com.ctuconnect.repository.BatchRepository;
import com.ctuconnect.repository.GenderRepository;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.kafka.core.KafkaTemplate;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private MajorRepository majorRepository;

    @Mock
    private BatchRepository batchRepository;

    @Mock
    private GenderRepository genderRepository;

    @Mock
    private UserMapper userMapper;

    @Mock
    private KafkaTemplate<String, Object> kafkaTemplate;

    @InjectMocks
    private UserService userService;

    private UserEntity testUser;
    private UserProfileDTO testUserProfileDTO;
    private UserRepository.UserProfileProjection testProjection;

    @BeforeEach
    void setUp() {
        testUser = UserEntity.builder()
                .id("test-user-id")
                .email("test@example.com")
                .username("testuser")
                .fullName("Test User")
                .role("USER")
                .isActive(true)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();

        testUserProfileDTO = UserProfileDTO.builder()
                .id("test-user-id")
                .email("test@example.com")
                .username("testuser")
                .fullName("Test User")
                .role("USER")
                .isActive(true)
                .build();

        testProjection = mock(UserRepository.UserProfileProjection.class);
        when(testProjection.getUser()).thenReturn(testUser);
        when(testProjection.getCollege()).thenReturn("Test College");
        when(testProjection.getFaculty()).thenReturn("Test Faculty");
        when(testProjection.getMajor()).thenReturn("Test Major");
        when(testProjection.getBatch()).thenReturn(2023);
        when(testProjection.getGender()).thenReturn("Male");
        when(testProjection.getFriendsCount()).thenReturn(5L);
        when(testProjection.getSentRequestsCount()).thenReturn(2L);
        when(testProjection.getReceivedRequestsCount()).thenReturn(3L);
    }

    @Test
    void getUserProfile_WhenUserExists_ShouldReturnUserProfile() {
        // Arrange
        when(userRepository.findUserProfileById("test-user-id"))
                .thenReturn(Optional.of(testProjection));
        when(userMapper.toUserProfileDTO(testProjection))
                .thenReturn(testUserProfileDTO);

        // Act
        UserProfileDTO result = userService.getUserProfile("test-user-id");

        // Assert
        assertNotNull(result);
        assertEquals("test-user-id", result.getId());
        assertEquals("test@example.com", result.getEmail());
        verify(userRepository).findUserProfileById("test-user-id");
        verify(userMapper).toUserProfileDTO(testProjection);
    }

    @Test
    void getUserProfile_WhenUserNotExists_ShouldThrowUserNotFoundException() {
        // Arrange
        when(userRepository.findUserProfileById("non-existent-id"))
                .thenReturn(Optional.empty());

        // Act & Assert
        assertThrows(UserNotFoundException.class, () -> {
            userService.getUserProfile("non-existent-id");
        });
        verify(userRepository).findUserProfileById("non-existent-id");
        verifyNoInteractions(userMapper);
    }

    @Test
    void createUser_WhenValidInput_ShouldCreateUser() {
        // Arrange
        when(userRepository.existsById("auth-user-id")).thenReturn(false);
        when(userRepository.existsByEmail("test@example.com")).thenReturn(false);
        when(userRepository.save(any(UserEntity.class))).thenReturn(testUser);

        // Act
        UserEntity result = userService.createUser("auth-user-id", "test@example.com", "testuser", "USER");

        // Assert
        assertNotNull(result);
        verify(userRepository).existsById("auth-user-id");
        verify(userRepository).existsByEmail("test@example.com");
        verify(userRepository).save(any(UserEntity.class));
        verify(kafkaTemplate).send(anyString(), anyString(), any());
    }

    @Test
    void updateUserProfile_WhenUserExists_ShouldUpdateProfile() {
        // Arrange
        UserUpdateDTO updateDTO = UserUpdateDTO.builder()
                .fullName("Updated Name")
                .bio("Updated bio")
                .build();

        when(userRepository.findById("test-user-id")).thenReturn(Optional.of(testUser));
        when(userRepository.save(any(UserEntity.class))).thenReturn(testUser);
        when(userRepository.findUserProfileById("test-user-id")).thenReturn(Optional.of(testProjection));
        when(userMapper.toUserProfileDTO(testProjection)).thenReturn(testUserProfileDTO);

        // Act
        UserProfileDTO result = userService.updateUserProfile("test-user-id", updateDTO);

        // Assert
        assertNotNull(result);
        verify(userRepository).findById("test-user-id");
        verify(userRepository).save(any(UserEntity.class));
        verify(kafkaTemplate).send(anyString(), anyString(), any());
    }

    @Test
    void sendFriendRequest_WhenValidUsers_ShouldSendRequest() {
        // Arrange
        UserEntity sender = UserEntity.builder().id("sender-id").isActive(true).build();
        UserEntity receiver = UserEntity.builder().id("receiver-id").isActive(true).build();

        when(userRepository.findById("sender-id")).thenReturn(Optional.of(sender));
        when(userRepository.findById("receiver-id")).thenReturn(Optional.of(receiver));
        when(userRepository.sendFriendRequest("sender-id", "receiver-id")).thenReturn(true);

        // Act
        assertDoesNotThrow(() -> {
            userService.sendFriendRequest("sender-id", "receiver-id");
        });

        // Assert
        verify(userRepository).findById("sender-id");
        verify(userRepository).findById("receiver-id");
        verify(userRepository).sendFriendRequest("sender-id", "receiver-id");
    }

    @Test
    void userExists_WhenUserExists_ShouldReturnTrue() {
        // Arrange
        when(userRepository.existsById("test-user-id")).thenReturn(true);

        // Act
        boolean result = userService.userExists("test-user-id");

        // Assert
        assertTrue(result);
        verify(userRepository).existsById("test-user-id");
    }

    @Test
    void userExists_WhenUserNotExists_ShouldReturnFalse() {
        // Arrange
        when(userRepository.existsById("non-existent-id")).thenReturn(false);

        // Act
        boolean result = userService.userExists("non-existent-id");

        // Assert
        assertFalse(result);
        verify(userRepository).existsById("non-existent-id");
    }
}
