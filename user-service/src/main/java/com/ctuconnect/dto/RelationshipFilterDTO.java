package com.ctuconnect.dto;

import lombok.Data;

@Data
public class RelationshipFilterDTO {
    private boolean isFriend = false;
    private boolean isSameCollege = false;
    private boolean isSameFaculty = false;
    private boolean isSameMajor = false;
    private boolean isSameBatch = false;

    // Additional options
    private Integer page = 0;
    private Integer size = 20;
}
