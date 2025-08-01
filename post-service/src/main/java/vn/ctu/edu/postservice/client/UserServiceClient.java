package vn.ctu.edu.postservice.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import vn.ctu.edu.postservice.dto.AuthorInfo;

@FeignClient(name = "user-service", url = "${user-service.url:http://localhost:8080}")
public interface UserServiceClient {
    @GetMapping("/api/users/authors/{id}")
    AuthorInfo getAuthorInfo(@PathVariable("id") String authorId);


}
