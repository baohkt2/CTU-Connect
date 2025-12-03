package com.ctuconnect.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import com.ctuconnect.dto.MediaResponse;
import com.ctuconnect.entity.Media;
import com.ctuconnect.security.SecurityContextHolder;
import com.ctuconnect.service.MediaService;
import java.util.UUID;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/media")
@RequiredArgsConstructor
@Slf4j
public class MediaController {

    private final MediaService mediaService;

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<MediaResponse> uploadFile(
            @RequestParam("file") MultipartFile file,

            @RequestParam(value = "description", required = false) String description) {

        String uploadedBy = SecurityContextHolder.getCurrentUserIdOrThrow();
        log.info("Received upload request for file: {} by user: {}", file.getOriginalFilename(), uploadedBy);
        MediaResponse response = mediaService.uploadFile(file, uploadedBy, description);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/{id}")
    public ResponseEntity<MediaResponse> getMediaById(@PathVariable UUID id) {
        log.info("Retrieving media with ID: {}", id);
        MediaResponse response = mediaService.getMediaById(id);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/me")
    public ResponseEntity<List<MediaResponse>> getMyMedia() {
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        log.info("Retrieving media for user: {}", userId);
        List<MediaResponse> responses = mediaService.getMediaByUser(userId);
        return ResponseEntity.ok(responses);
    }

    @GetMapping("/user/{uploadedBy}")
    public ResponseEntity<List<MediaResponse>> getMediaByUser(@PathVariable String uploadedBy) {
        log.info("Retrieving media for user: {}", uploadedBy);
        List<MediaResponse> responses = mediaService.getMediaByUser(uploadedBy);
        return ResponseEntity.ok(responses);
    }

    @GetMapping("/type/{mediaType}")
    public ResponseEntity<List<MediaResponse>> getMediaByType(@PathVariable String mediaType) {
        log.info("Retrieving media of type: {}", mediaType);
        Media.MediaType type = Media.MediaType.valueOf(mediaType.toUpperCase());
        List<MediaResponse> responses = mediaService.getMediaByType(type);
        return ResponseEntity.ok(responses);
    }

    @GetMapping("/search")
    public ResponseEntity<List<MediaResponse>> searchMedia(@RequestParam String keyword) {
        log.info("Searching media with keyword: {}", keyword);
        List<MediaResponse> responses = mediaService.searchMedia(keyword);
        return ResponseEntity.ok(responses);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Map<String, String>> deleteMedia(@PathVariable UUID id) {
        log.info("Deleting media with ID: {}", id);
        mediaService.deleteMedia(id);

        Map<String, String> response = new HashMap<>();
        response.put("message", "Media deleted successfully");
        response.put("mediaId", id.toString());

        return ResponseEntity.ok(response);
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> healthCheck() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "UP");
        response.put("service", "media-service");
        return ResponseEntity.ok(response);
    }
}
