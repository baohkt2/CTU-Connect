package vn.ctu.edu.postservice.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@FeignClient(name = "media-service", url = "${media-service.url:http://localhost:8080}")
public interface MediaServiceClient {

    @PostMapping(value = "/api/media/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    MediaUploadResponse uploadFile(@RequestPart("file") MultipartFile file,
                                  @RequestParam("type") String type);

    @DeleteMapping("/api/media/{id}")
    void deleteFile(@PathVariable("id") String fileId);

    @GetMapping("/api/media/{id}")
    MediaResponse getFileInfo(@PathVariable("id") String fileId);

    // Response DTOs for media service
    class MediaUploadResponse {
        private String id;
        private String fileName;
        private String fileUrl;
        private String fileType;
        private long fileSize;

        // Getters and Setters
        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getFileName() {
            return fileName;
        }

        public void setFileName(String fileName) {
            this.fileName = fileName;
        }

        public String getFileUrl() {
            return fileUrl;
        }

        public void setFileUrl(String fileUrl) {
            this.fileUrl = fileUrl;
        }

        public String getFileType() {
            return fileType;
        }

        public void setFileType(String fileType) {
            this.fileType = fileType;
        }

        public long getFileSize() {
            return fileSize;
        }

        public void setFileSize(long fileSize) {
            this.fileSize = fileSize;
        }
    }

    class MediaResponse {
        private String id;
        private String fileName;
        private String fileUrl;
        private String fileType;
        private long fileSize;

        // Getters and Setters
        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getFileName() {
            return fileName;
        }

        public void setFileName(String fileName) {
            this.fileName = fileName;
        }

        public String getFileUrl() {
            return fileUrl;
        }

        public void setFileUrl(String fileUrl) {
            this.fileUrl = fileUrl;
        }

        public String getFileType() {
            return fileType;
        }

        public void setFileType(String fileType) {
            this.fileType = fileType;
        }

        public long getFileSize() {
            return fileSize;
        }

        public void setFileSize(long fileSize) {
            this.fileSize = fileSize;
        }
    }
}
