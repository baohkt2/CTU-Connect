package com.ctuconnect.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import com.ctuconnect.entity.Media;

@Service
@RequiredArgsConstructor
@Slf4j
public class KafkaProducerService {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    private static final String MEDIA_UPLOAD_TOPIC = "media-upload-events";
    private static final String MEDIA_DELETE_TOPIC = "media-delete-events";

    public void sendMediaUploadEvent(Media media) {
        try {
            MediaEvent event = new MediaEvent(
                media.getIdAsString(),
                media.getCloudinaryUrl(),
                media.getMediaType().toString(),
                media.getUploadedBy(),
                "UPLOADED"
            );

            kafkaTemplate.send(MEDIA_UPLOAD_TOPIC, event);
            log.info("Sent media upload event for media ID: {}", media.getId());
        } catch (Exception e) {
            log.error("Failed to send media upload event: {}", e.getMessage());
        }
    }

    public void sendMediaDeleteEvent(Media media) {
        try {
            MediaEvent event = new MediaEvent(
                media.getIdAsString(),
                media.getCloudinaryUrl(),
                media.getMediaType().toString(),
                media.getUploadedBy(),
                "DELETED"
            );

            kafkaTemplate.send(MEDIA_DELETE_TOPIC, event);
            log.info("Sent media delete event for media ID: {}", media.getId());
        } catch (Exception e) {
            log.error("Failed to send media delete event: {}", e.getMessage());
        }
    }

    public static class MediaEvent {
        public String mediaId;
        public String url;
        public String mediaType;
        public String uploadedBy;
        public String action;

        public MediaEvent(String mediaId, String url, String mediaType, String uploadedBy, String action) {
            this.mediaId = mediaId;
            this.url = url;
            this.mediaType = mediaType;
            this.uploadedBy = uploadedBy;
            this.action = action;
        }
    }
}
