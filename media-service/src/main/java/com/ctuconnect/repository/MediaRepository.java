package com.ctuconnect.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import com.ctuconnect.entity.Media;
import java.util.UUID;

import java.util.List;
import java.util.Optional;

@Repository
public interface MediaRepository extends JpaRepository<Media, UUID> {

    List<Media> findByUploadedBy(String uploadedBy);

    List<Media> findByMediaType(Media.MediaType mediaType);

    List<Media> findByUploadedByAndMediaType(String uploadedBy, Media.MediaType mediaType);

    Optional<Media> findByCloudinaryPublicId(String publicId);

    @Query("SELECT m FROM Media m WHERE m.fileName LIKE %:keyword% OR m.originalFileName LIKE %:keyword%")
    List<Media> findByFileNameContaining(@Param("keyword") String keyword);

    boolean existsByCloudinaryPublicId(String publicId);
}
