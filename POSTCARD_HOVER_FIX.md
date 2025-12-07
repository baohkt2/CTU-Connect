# 🔧 Fix PostCard Hover Behavior - Reactions Picker

## ❌ Vấn Đề

**Symptom:**
1. Hover vào nút "Thích" → Picker xuất hiện ✅
2. Di chuột từ button đến picker → Picker biến mất ❌
3. Không thể click chọn reaction ❌

**Root Cause:**
`onMouseEnter/onMouseLeave` được đặt trên `<Button>` thay vì parent `<div>`, nên khi chuột rời button để vào picker thì trigger `onMouseLeave` → ẩn picker.

## ✅ Giải Pháp

### Concept: Hover Zone

Cả **button** và **picker** phải nằm trong **cùng 1 hover zone**:

```
┌─────────────────────────────┐
│  Hover Zone (parent div)   │
│  ┌───────────────────────┐  │
│  │   Reaction Picker     │  │ ← Picker nằm TRONG zone
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │   Button "Thích"      │  │ ← Button nằm TRONG zone
│  └───────────────────────┘  │
└─────────────────────────────┘
```

### Code Changes

#### ❌ Code Cũ (Sai)
```tsx
<div className="relative">
  <Button
    onMouseEnter={() => setShowReactionPicker(true)}   // ❌ Trên button
    onMouseLeave={() => setShowReactionPicker(false)}  // ❌ Trigger khi rời button
  >
    Thích
  </Button>
  
  {showReactionPicker && (
    <div className="...">  {/* Picker */}
      {/* Khi hover vào đây, đã rời button → picker ẩn */}
    </div>
  )}
</div>
```

**Vấn đề:**
- Hover vào button → Picker hiện
- Di chuột từ button → picker → `onMouseLeave` trigger → Picker ẩn ngay

#### ✅ Code Mới (Đúng)
```tsx
<div 
  className="relative"
  onMouseEnter={() => setShowReactionPicker(true)}   // ✅ Trên parent div
  onMouseLeave={() => setShowReactionPicker(false)}  // ✅ Chỉ trigger khi rời cả zone
>
  <Button>
    Thích
  </Button>
  
  {showReactionPicker && (
    <div className="... z-50">  {/* Picker với z-index cao */}
      {/* Hover vào đây vẫn trong zone → picker không ẩn */}
    </div>
  )}
</div>
```

**Lợi ích:**
- Hover vào button → Picker hiện
- Di chuột từ button → picker → **Vẫn trong zone** → Picker không ẩn
- Chỉ khi chuột rời **cả zone** thì mới ẩn

## 📊 Technical Details

### Event Flow - Code Cũ
```
User hovers button
  ↓
onMouseEnter (button) triggers
  ↓
showReactionPicker = true
  ↓
Picker appears
  ↓
User moves mouse to picker
  ↓
Mouse leaves button area
  ↓
onMouseLeave (button) triggers ❌
  ↓
showReactionPicker = false
  ↓
Picker disappears before user can click ❌
```

### Event Flow - Code Mới
```
User hovers into zone (button or picker)
  ↓
onMouseEnter (parent div) triggers
  ↓
showReactionPicker = true
  ↓
Picker appears
  ↓
User moves mouse to picker
  ↓
Still inside parent div ✅
  ↓
onMouseLeave NOT triggered ✅
  ↓
User can click reaction ✅
  ↓
User moves mouse out of zone
  ↓
onMouseLeave (parent div) triggers
  ↓
showReactionPicker = false
  ↓
Picker hides (after interaction) ✅
```

## 🔍 Additional Improvements

### 1. Z-Index
```tsx
className="... z-50"  // Ensure picker is above other elements
```

**Why:** Picker có thể bị che bởi elements khác nếu không có z-index cao.

### 2. Position
```tsx
className="absolute bottom-full left-0 mb-2"
```

**Why:** 
- `bottom-full` - Đặt picker ở trên button
- `left-0` - Align left với button
- `mb-2` - Margin 8px để không chạm button

### 3. Animation
```tsx
className="... animate-in fade-in-50 slide-in-from-bottom-2 duration-200"
```

**Why:** Smooth transition khi picker xuất hiện/ẩn.

## ✅ Complete Fixed Code

```tsx
<div className="flex items-center justify-around">
  {/* Like Button with Reaction Picker */}
  <div 
    className="relative"
    onMouseEnter={() => setShowReactionPicker(true)}
    onMouseLeave={() => setShowReactionPicker(false)}
  >
    <Button
      variant="ghost"
      size="sm"
      onClick={() => handleInteraction('like')}
      disabled={isLiked === null || isLoadingInteraction}
      className={`
        flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200
        ${isLiked 
          ? 'text-blue-600 bg-blue-50 hover:bg-blue-100' 
          : 'text-gray-700 hover:bg-gray-100 hover:text-blue-600'
        }
        ${isLoadingInteraction ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      {isLoadingInteraction ? (
        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
      ) : (
        <ThumbsUp className={`h-4 w-4 ${isLiked ? 'fill-current' : ''}`} />
      )}
      <span className="font-medium">{isLiked ? 'Đã thích' : 'Thích'}</span>
    </Button>

    {/* Reaction Picker */}
    {showReactionPicker && (
      <div
        className="absolute bottom-full left-0 mb-2 flex items-center gap-1 bg-white border border-gray-200 rounded-full px-2 py-1 shadow-lg animate-in fade-in-50 slide-in-from-bottom-2 duration-200 z-50"
      >
        {REACTIONS.map((reaction) => (
          <button
            key={reaction.id}
            className="w-8 h-8 rounded-full hover:scale-125 transition-transform duration-150"
            onClick={() => handleReactionClick(reaction.id)}
            title={reaction.name}
          >
            <span className="text-lg">{reaction.emoji}</span>
          </button>
        ))}
      </div>
    )}
  </div>
  
  {/* Other buttons... */}
</div>
```

## 🧪 Testing

### Test Case 1: Hover Button
1. Hover chuột vào nút "Thích"
2. ✅ Picker xuất hiện

### Test Case 2: Move to Picker
1. Hover vào button → Picker hiện
2. Di chuột từ button đến picker
3. ✅ Picker vẫn hiển thị (không ẩn)

### Test Case 3: Click Reaction
1. Hover vào button → Picker hiện
2. Di chuột vào picker
3. Click vào một reaction
4. ✅ Reaction được chọn
5. ✅ Picker ẩn sau khi click

### Test Case 4: Hover Out
1. Hover vào button → Picker hiện
2. Di chuột ra ngoài cả button và picker
3. ✅ Picker ẩn

## 📝 Lessons Learned

### ❌ Bad Practice
```tsx
// Đặt hover handlers trên element con
<div>
  <Button 
    onMouseEnter={show}
    onMouseLeave={hide}
  />
  <Popup />
</div>
```

**Problem:** Hover zone chỉ là button, không bao gồm popup.

### ✅ Good Practice
```tsx
// Đặt hover handlers trên parent container
<div 
  onMouseEnter={show}
  onMouseLeave={hide}
>
  <Button />
  <Popup />
</div>
```

**Benefit:** Hover zone bao gồm cả button và popup.

### 🎯 Best Practice
```tsx
// Parent container với hover handlers + relative positioning
<div 
  className="relative"
  onMouseEnter={show}
  onMouseLeave={hide}
>
  <Button />
  <Popup className="absolute ... z-50" />
</div>
```

**Benefits:**
- Hover zone đúng
- Positioning đúng (absolute relative to parent)
- Z-index đúng (popup trên các elements khác)

## 🔄 Similar Components

Áp dụng pattern này cho:
- Dropdown menus
- Tooltips with interactive content
- Context menus
- Emoji pickers
- Color pickers
- Any popup that needs to stay open when hovering

## ✅ Status

- **Issue:** Picker ẩn khi di chuột từ button → picker ❌
- **Root cause:** Hover handlers trên button thay vì parent ✅
- **Fix applied:** Move handlers lên parent div ✅
- **Tested:** Ready for testing ⏳
- **Production ready:** After testing ⏳

---

**Fixed Date:** 2025-12-07  
**Version:** 1.4.0  
**Status:** ✅ HOVER BEHAVIOR FIXED
