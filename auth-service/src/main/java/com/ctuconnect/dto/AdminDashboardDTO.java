package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AdminDashboardDTO {
    private long totalUsers;
    private long activeUsers;
    private long inactiveUsers;
    private long unverifiedUsers;
    private long newUsersToday;
    private long newUsersThisWeek;
    private long newUsersThisMonth;
    private Map<String, Long> usersByRole;
    private List<UserManagementDTO> recentUsers;
    private Map<String, Long> loginStatistics;
}
