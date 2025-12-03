package com.ctuconnect.client;

import com.ctuconnect.dto.UserDTO;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

@FeignClient(name = "user-service")
public interface UserServiceClient {

    @PostMapping("/api/users/register")
    UserDTO createUser(@RequestBody UserDTO userDTO);
}