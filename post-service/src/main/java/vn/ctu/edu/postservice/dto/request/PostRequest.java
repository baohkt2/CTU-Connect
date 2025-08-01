package vn.ctu.edu.postservice.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class PostRequest {
    @NotBlank(message = "Content is required")
    @Size(max = 5000, message = "Content cannot exceed 5000 characters")
    private String content;


    private List<String> tags = new ArrayList<>();

    private String category;


}
