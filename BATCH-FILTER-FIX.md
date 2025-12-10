# Batch Filter Fix - December 10, 2025

## Problem
Batch entity chỉ có `year` (String), không có `name` property. Frontend expects both `year` và `name`.

## Root Cause

**BatchEntity:**
```java
@Node("Batch")
public class BatchEntity {
    @Id
    private String year;  // Only property
    private String description;
}
```

**BatchInfo DTO:**
```java
public static class BatchInfo {
    private String year;  // Only field
}
```

**Frontend Expected:**
```typescript
interface BatchOption {
  year: number;  // ❌ Expected number, got string
  name: string;  // ❌ Expected name, doesn't exist
}
```

## Solution

### Backend - Added getName() Helper
**File:** `CategoryDTO.java`

```java
public static class BatchInfo {
    private String year;
    
    // Helper method to get display name (K47, K48, etc.)
    public String getName() {
        if (year == null || year.isEmpty()) {
            return "";
        }
        try {
            int yearNum = Integer.parseInt(year);
            int k = yearNum - 1974; // CTU K1 started 1974
            return "K" + k + " (" + year + ")";
        } catch (NumberFormatException e) {
            return year;
        }
    }
}
```

### Frontend - Updated Type & Usage
**File:** `FriendSuggestions.tsx`

**Type Updated:**
```typescript
interface BatchOption {
  year: string;   // ✅ Changed from number to string
  name?: string;  // ✅ Optional, can be generated
}
```

**Dropdown Updated:**
```tsx
{batches.map((batch: any) => (
  <option key={batch.year} value={batch.year}>
    {batch.name || batch.year}  // Use name if exists, fallback to year
  </option>
))}
```

## How It Works

### API Response
```json
[
  { "year": "2024", "name": "K50 (2024)" },
  { "year": "2023", "name": "K49 (2023)" },
  { "year": "2022", "name": "K48 (2022)" }
]
```

### Calculation Logic
- K number = year - 1974
- K50 = 2024 - 1974 = 50
- K49 = 2023 - 1974 = 49

### Display
- **Dropdown shows:** "K50 (2024)"
- **Value saved:** "2024"
- **Filter sent:** `batch=2024`

## Database Structure

**Batches stored as:**
```cypher
(:Batch {year: "2024"})
(:Batch {year: "2023"})
(:Batch {year: "2022"})
```

**Users linked to batch:**
```cypher
(:User)-[:IN_BATCH]->(:Batch {year: "2024"})
```

## Search Query

When user selects "K50 (2024)":
```
GET /api/users/friend-suggestions/search?batch=2024
```

Backend matches:
```cypher
MATCH (u:User)-[:IN_BATCH]->(:Batch {year: $batch})
WHERE $batch = "2024"
```

## Benefits

✅ **No DB changes** - Works with existing batch structure  
✅ **Backward compatible** - Frontend handles missing name  
✅ **User-friendly** - Shows "K50 (2024)" instead of just "2024"  
✅ **Flexible** - getName() method can be enhanced later  

## Files Modified

1. **user-service/.../CategoryDTO.java** - Added getName() helper
2. **client-frontend/.../FriendSuggestions.tsx** - Updated type to string

## Result

**Batch filters giờ hoạt động chính xác với year as String!** ✅
