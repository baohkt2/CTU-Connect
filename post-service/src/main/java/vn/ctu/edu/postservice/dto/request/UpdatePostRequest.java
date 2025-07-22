package vn.ctu.edu.postservice.dto.request;

import jakarta.validation.constraints.Size;

import java.util.List;

public class UpdatePostRequest {

    @Size(max = 200, message = "Title cannot exceed 200 characters")
    private String title;

    @Size(max = 5000, message = "Content cannot exceed 5000 characters")
    private String content;

    private List<String> tags;

    private String category;

    // Getters and Setters
    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public List<String> getTags() {
        return tags;
    }

    public void setTags(List<String> tags) {
        this.tags = tags;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }
}
