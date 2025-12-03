package com.ctuconnect.service;

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import com.ctuconnect.dto.MediaResponse;
import com.ctuconnect.entity.Media;
import com.ctuconnect.exception.MediaNotFoundException;
import com.ctuconnect.exception.MediaUploadException;
import com.ctuconnect.repository.MediaRepository;


import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class MediaService {

    private final MediaRepository mediaRepository;
    private final Cloudinary cloudinary;
    private final KafkaProducerService kafkaProducerService;

    public MediaResponse uploadFile(MultipartFile file, String uploadedBy, String description) {
        try {
            log.info("Starting upload for file: {} by user: {}", file.getOriginalFilename(), uploadedBy);

            // Validate file
            validateFile(file);

            // Determine media type
            Media.MediaType mediaType = determineMediaType(file.getContentType());

            // Prepare upload options based on media type
            Map<String, Object> uploadOptions = ObjectUtils.asMap(
                "resource_type", getCloudinaryResourceType(mediaType),
                "folder", "ctu-connect/" + mediaType.toString().toLowerCase(),
                "public_id", generatePublicId(file.getOriginalFilename(), uploadedBy)
            );

            // Special handling for PDFs and documents to ensure browser compatibility
            if (mediaType == Media.MediaType.DOCUMENT) {
                uploadOptions.put("flags", "attachment");
                uploadOptions.put("format", "pdf");
                if (file.getContentType().contains("pdf")) {
                    // For PDFs, we want them to be viewable in browser
                    uploadOptions.remove("flags"); // Remove attachment flag to allow inline viewing
                }
            }

            // Upload to Cloudinary
            Map<?, ?> uploadResult = cloudinary.uploader().upload(file.getBytes(), uploadOptions);

            // Save metadata to database
            Media media = new Media();
            media.setFileName((String) uploadResult.get("public_id"));
            media.setOriginalFileName(file.getOriginalFilename());
            media.setCloudinaryUrl((String) uploadResult.get("secure_url"));
            media.setCloudinaryPublicId((String) uploadResult.get("public_id"));
            media.setContentType(file.getContentType());
            media.setMediaType(mediaType);
            media.setFileSize(file.getSize());
            media.setDescription(description);
            media.setUploadedBy(uploadedBy);

            Media savedMedia = mediaRepository.save(media);

            // Send Kafka event
            kafkaProducerService.sendMediaUploadEvent(savedMedia);

            log.info("File uploaded successfully: {}", savedMedia.getId());
            return convertToResponse(savedMedia);

        } catch (IOException e) {
            log.error("Error uploading file: {}", e.getMessage());
            throw new MediaUploadException("Failed to upload file: " + e.getMessage());
        }
    }

    public MediaResponse getMediaById(UUID id) {
        
        Media media = mediaRepository.findById(id)
            .orElseThrow(() -> new MediaNotFoundException("Media not found with id: " + id));
        return convertToResponse(media);
    }

    public List<MediaResponse> getMediaByUser(String uploadedBy) {
        List<Media> mediaList = mediaRepository.findByUploadedBy(uploadedBy);
        return mediaList.stream()
            .map(this::convertToResponse)
            .collect(Collectors.toList());
    }

    public List<MediaResponse> getMediaByType(Media.MediaType mediaType) {
        List<Media> mediaList = mediaRepository.findByMediaType(mediaType);
        return mediaList.stream()
            .map(this::convertToResponse)
            .collect(Collectors.toList());
    }

    public List<MediaResponse> searchMedia(String keyword) {
        List<Media> mediaList = mediaRepository.findByFileNameContaining(keyword);
        return mediaList.stream()
            .map(this::convertToResponse)
            .collect(Collectors.toList());
    }

    public void deleteMedia(UUID id) {
        Media media = mediaRepository.findById(id)
            .orElseThrow(() -> new MediaNotFoundException("Media not found with id: " + id));

        try {
            // Delete from Cloudinary
            cloudinary.uploader().destroy(media.getCloudinaryPublicId(),
                ObjectUtils.asMap("resource_type", getCloudinaryResourceType(media.getMediaType())));

            // Delete from database
            mediaRepository.delete(media);

            // Send Kafka event
            kafkaProducerService.sendMediaDeleteEvent(media);

            log.info("Media deleted successfully: {}", id);

        } catch (IOException e) {
            log.error("Error deleting file from Cloudinary: {}", e.getMessage());
            throw new MediaUploadException("Failed to delete file: " + e.getMessage());
        }
    }

    private void validateFile(MultipartFile file) {
        if (file.isEmpty()) {
            throw new MediaUploadException("File is empty");
        }

        // Check file size (10MB limit)
        if (file.getSize() > 10 * 1024 * 1024) {
            throw new MediaUploadException("File size exceeds 10MB limit");
        }

        // Check content type
        String contentType = file.getContentType();
        if (contentType == null) {
            throw new MediaUploadException("File content type is not supported");
        }
    }

    private Media.MediaType determineMediaType(String contentType) {
        if (contentType.startsWith("image/")) {
            return Media.MediaType.IMAGE;
        } else if (contentType.startsWith("video/")) {
            return Media.MediaType.VIDEO;
        } else if (contentType.startsWith("audio/")) {
            return Media.MediaType.AUDIO;
        } else {
            return Media.MediaType.DOCUMENT;
        }
    }

    private String getCloudinaryResourceType(Media.MediaType mediaType) {
        return switch (mediaType) {
            case IMAGE -> "image";
            case VIDEO -> "video";
            case AUDIO -> "video"; // Cloudinary uses 'video' for audio files
            case DOCUMENT -> "raw";
        };
    }

    private String generatePublicId(String originalFileName, String uploadedBy) {
        String fileName = originalFileName.replaceAll("[^a-zA-Z0-9._-]", "_");
        return uploadedBy + "_" + System.currentTimeMillis() + "_" + fileName;
    }

    private MediaResponse convertToResponse(Media media) {
        return new MediaResponse(
            media.getIdAsString(),
            media.getFileName(),
            media.getOriginalFileName(),
            media.getCloudinaryUrl(),
            media.getCloudinaryPublicId(),
            media.getContentType(),
            media.getMediaType(),
            media.getFileSize(),
            media.getDescription(),
            media.getUploadedBy(),
            media.getCreatedAt(),
            media.getUpdatedAt()
        );
    }
}
