package com.ctuconnect.dto;


import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor

public class BatchInfo {
    private String id;
    private String year;

    public BatchInfo(String id, String year) {
        this.id = id;
        this.year = year;
    }
}
