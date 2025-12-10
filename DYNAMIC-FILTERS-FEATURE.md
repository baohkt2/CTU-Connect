# Dynamic Filters from Database - December 10, 2025

## Problem
Filter dropdowns (Faculty, Batch, College) sử dụng hardcoded values thay vì load từ database.

## Solution
Load categories từ backend API `/api/users/categories/*`.

## Changes Made

### Backend APIs (Already Available) ✅
```
GET /api/users/categories/colleges  → List of colleges
GET /api/users/categories/faculties → List of faculties  
GET /api/users/categories/batches   → List of batches
GET /api/users/categories/majors    → List of majors
GET /api/users/categories/all       → Hierarchical structure
```

### Frontend Service Updated

**File:** `client-frontend/src/services/userService.ts`

**Added methods:**
```typescript
// Get all colleges with their faculties and majors
async getColleges(): Promise<any[]> {
  const response = await api.get('/users/categories/colleges');
  return response.data;
}

// Get all faculties with their majors
async getFaculties(): Promise<any[]> {
  const response = await api.get('/users/categories/faculties');
  return response.data;
}

// Get all batches
async getBatches(): Promise<any[]> {
  const response = await api.get('/users/categories/batches');
  return response.data;
}

// Get all majors
async getMajors(): Promise<any[]> {
  const response = await api.get('/users/categories/majors');
  return response.data;
}

// Get all categories in hierarchical structure
async getAllCategories(): Promise<any> {
  const response = await api.get('/users/categories/all');
  return response.data;
}
```

### FriendSuggestions Component Updated

**File:** `client-frontend/src/features/users/components/friends/FriendSuggestions.tsx`

**Added:**
```typescript
// Type definitions
interface CollegeOption {
  code: string;
  name: string;
}

interface FacultyOption {
  code: string;
  name: string;
}

interface BatchOption {
  year: number;
  name: string;
}

// State for categories
const [colleges, setColleges] = useState<CollegeOption[]>([]);
const [faculties, setFaculties] = useState<FacultyOption[]>([]);
const [batches, setBatches] = useState<BatchOption[]>([]);
const [loadingCategories, setLoadingCategories] = useState(true);

// Load categories on mount
useEffect(() => {
  loadCategories();
  loadSuggestions();
}, []);

// Fetch categories from API
const loadCategories = async () => {
  try {
    setLoadingCategories(true);
    const [collegesData, facultiesData, batchesData] = await Promise.all([
      userService.getColleges(),
      userService.getFaculties(),
      userService.getBatches()
    ]);
    
    setColleges(collegesData || []);
    setFaculties(facultiesData || []);
    setBatches(batchesData || []);
  } catch (err) {
    console.error('Error loading categories:', err);
    // Don't show error to user, just use empty arrays
  } finally {
    setLoadingCategories(false);
  }
};
```

**Updated dropdowns:**
```tsx
{/* Faculty Dropdown */}
<select
  value={filters.faculty}
  onChange={(e) => setFilters({...filters, faculty: e.target.value})}
  disabled={loadingCategories}
>
  <option value="">All Faculties</option>
  {faculties.map((faculty: any) => (
    <option key={faculty.code} value={faculty.name}>
      {faculty.name}
    </option>
  ))}
</select>

{/* Batch Dropdown */}
<select
  value={filters.batch}
  onChange={(e) => setFilters({...filters, batch: e.target.value})}
  disabled={loadingCategories}
>
  <option value="">All Batches</option>
  {batches.map((batch: any) => (
    <option key={batch.year} value={batch.year.toString()}>
      {batch.name}
    </option>
  ))}
</select>

{/* College Dropdown */}
<select
  value={filters.college}
  onChange={(e) => setFilters({...filters, college: e.target.value})}
  disabled={loadingCategories}
>
  <option value="">All Colleges</option>
  {colleges.map((college: any) => (
    <option key={college.code} value={college.name}>
      {college.name}
    </option>
  ))}
</select>
```

## Before vs After

### Before (Hardcoded) ❌
```tsx
<select>
  <option value="">All Faculties</option>
  <option value="Công nghệ thông tin">Công nghệ thông tin</option>
  <option value="Kinh tế">Kinh tế</option>
  <option value="Nông nghiệp">Nông nghiệp</option>
  {/* Fixed list - not from database */}
</select>
```

### After (Dynamic) ✅
```tsx
<select disabled={loadingCategories}>
  <option value="">All Faculties</option>
  {faculties.map((faculty: any) => (
    <option key={faculty.code} value={faculty.name}>
      {faculty.name}
    </option>
  ))}
  {/* Loaded from database */}
</select>
```

## Features

1. **Parallel Loading** - Fetch all categories at once with `Promise.all()`
2. **Loading State** - Disable dropdowns while loading
3. **Error Handling** - Silent error handling, don't break UI
4. **Empty Fallback** - Use empty arrays if API fails
5. **Loading Indicator** - Show "Loading..." text under dropdowns

## Benefits

✅ **Always Up-to-Date** - Filter options match database  
✅ **No Manual Updates** - Add faculty/batch in database, it appears automatically  
✅ **Consistent Data** - Same source of truth for all components  
✅ **Better UX** - Shows actual available options  
✅ **Maintainable** - No hardcoded lists to update  

## API Response Format

**Colleges:**
```json
[
  { "code": "DHCT", "name": "Đại học Cần Thơ" },
  { "code": "KHTN", "name": "Khoa học Tự nhiên" }
]
```

**Faculties:**
```json
[
  { "code": "CNTT", "name": "Công nghệ thông tin", "majors": [...] },
  { "code": "KT", "name": "Kinh tế", "majors": [...] }
]
```

**Batches:**
```json
[
  { "year": 2024, "name": "K50 (2024)" },
  { "year": 2023, "name": "K49 (2023)" }
]
```

## Files Modified

1. **client-frontend/src/services/userService.ts**
   - Added getColleges()
   - Added getFaculties()
   - Added getBatches()
   - Added getMajors()
   - Added getAllCategories()

2. **client-frontend/src/features/users/components/friends/FriendSuggestions.tsx**
   - Added type interfaces
   - Added state for categories
   - Added loadCategories() function
   - Updated all filter dropdowns
   - Added loading states

## Result

**Filters giờ load 100% từ database!** 🎉
