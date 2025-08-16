package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
public class Info {
    private String id;
    private String name;
    private String code;

    public Info(String id, String name, String code) {
        this.id = id;
        this.name = name;
        this.code = code;
    }
}
