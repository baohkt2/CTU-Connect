# Hướng Dẫn Migration Dữ Liệu Reactions (Tùy Chọn)

## Tổng Quan

Tài liệu này hướng dẫn cách xử lý dữ liệu reactions cũ trong database nếu cần thiết. **Lưu ý:** Migration này là TÙY CHỌN và không bắt buộc để hệ thống hoạt động.

## Tình Huống

Sau khi cập nhật hệ thống reaction, trong database có thể tồn tại các reactions cũ với các giá trị:
- `LOVE` (❤️ Yêu thích)
- `HAHA` (😂 Haha)
- `WOW` (😮 Wow)
- `SAD` (😢 Buồn)
- `ANGRY` (😠 Phẫn nộ)

## Các Phương Án Xử Lý

### Phương Án 1: Giữ Nguyên (Khuyến Nghị)

**Ưu điểm:**
- Không mất dữ liệu lịch sử
- Không cần chạy migration script
- Backend vẫn xử lý được reactions cũ
- Phù hợp với hệ thống đang production

**Nhược điểm:**
- Reactions cũ không hiển thị trong UI mới
- Có thể gây nhầm lẫn nếu xem raw data

**Cách thực hiện:**
- Không làm gì cả! Hệ thống tự động bỏ qua reactions cũ trong UI

### Phương Án 2: Convert Sang Reactions Mới

Nếu muốn convert reactions cũ sang reactions mới, sử dụng mapping sau:

| Reaction Cũ | Reaction Mới Đề Xuất | Lý Do |
|-------------|----------------------|-------|
| `LOVE` → `LIKE` | 👍 Thích | Cả hai đều thể hiện sự yêu thích |
| `HAHA` → `INSIGHTFUL` | 💡 Sáng Suốt | Convert comment hài hước sang nội dung thú vị |
| `WOW` → `INSIGHTFUL` | 💡 Sáng Suốt | Wow thường dành cho nội dung bất ngờ/hay |
| `SAD` → `QUESTION` | ❓ Cần Thảo Luận | Chuyển cảm xúc tiêu cực sang discussion |
| `ANGRY` → `QUESTION` | ❓ Cần Thảo Luận | Chuyển cảm xúc tiêu cực sang discussion |

**MongoDB Migration Script:**

```javascript
// Connect to MongoDB
use post_db;

// Backup collection trước khi migrate
db.interactions.aggregate([
  { $match: { reactionType: { $in: ["LOVE", "HAHA", "WOW", "SAD", "ANGRY"] } } },
  { $out: "interactions_backup" }
]);

// Convert reactions
db.interactions.updateMany(
  { reactionType: "LOVE" },
  { $set: { reactionType: "LIKE" } }
);

db.interactions.updateMany(
  { reactionType: { $in: ["HAHA", "WOW"] } },
  { $set: { reactionType: "INSIGHTFUL" } }
);

db.interactions.updateMany(
  { reactionType: { $in: ["SAD", "ANGRY"] } },
  { $set: { reactionType: "QUESTION" } }
);

// Verify changes
db.interactions.aggregate([
  { $group: { _id: "$reactionType", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
]);
```

### Phương Án 3: Xóa Reactions Cũ

**Cảnh báo:** Phương án này sẽ mất dữ liệu vĩnh viễn!

```javascript
// Backup trước
db.interactions.aggregate([
  { $match: { reactionType: { $in: ["LOVE", "HAHA", "WOW", "SAD", "ANGRY"] } } },
  { $out: "interactions_deleted_backup" }
]);

// Xóa reactions cũ
db.interactions.deleteMany({
  reactionType: { $in: ["LOVE", "HAHA", "WOW", "SAD", "ANGRY"] }
});

// Verify
db.interactions.countDocuments({
  reactionType: { $in: ["LOVE", "HAHA", "WOW", "SAD", "ANGRY"] }
});
// Should return 0
```

## Kiểm Tra Dữ Liệu Hiện Tại

### Đếm Reactions Cũ

```javascript
use post_db;

// Đếm theo từng loại reaction
db.interactions.aggregate([
  { $match: { type: { $in: ["LIKE", "REACTION"] } } },
  { $group: { 
      _id: "$reactionType", 
      count: { $sum: 1 } 
  }},
  { $sort: { count: -1 } }
]);

// Tổng số reactions cũ
db.interactions.countDocuments({
  reactionType: { $in: ["LOVE", "HAHA", "WOW", "SAD", "ANGRY"] }
});
```

### Xem Sample Data

```javascript
// Xem 10 reactions cũ đầu tiên
db.interactions.find({
  reactionType: { $in: ["LOVE", "HAHA", "WOW", "SAD", "ANGRY"] }
}).limit(10).pretty();

// Xem 10 reactions mới đầu tiên
db.interactions.find({
  reactionType: { $in: ["LIKE", "INSIGHTFUL", "RELEVANT", "USEFUL_SOURCE", "QUESTION"] }
}).limit(10).pretty();
```

## Post Stats Update

Nếu đã convert reactions, cần cập nhật lại stats của posts:

```javascript
// Cập nhật reaction counts trong post stats
db.posts.find({}).forEach(function(post) {
  // Lấy tất cả reactions cho post này
  var reactionCounts = {};
  
  db.interactions.aggregate([
    { $match: { postId: post._id.toString(), type: { $in: ["LIKE", "REACTION"] } } },
    { $group: { _id: "$reactionType", count: { $sum: 1 } } }
  ]).forEach(function(result) {
    reactionCounts[result._id] = result.count;
  });
  
  // Update post stats
  db.posts.updateOne(
    { _id: post._id },
    { $set: { "stats.reactions": reactionCounts } }
  );
});
```

## Khuyến Nghị

**Cho môi trường Production:**
- ✅ Sử dụng Phương Án 1 (Giữ Nguyên)
- ✅ Chạy backup trước khi thực hiện bất kỳ migration nào
- ✅ Test migration trên staging environment trước

**Cho môi trường Development:**
- ✅ Có thể xóa reactions cũ để clean data
- ✅ Hoặc convert để test với data mới

## Rollback Plan

Nếu có vấn đề sau khi migration:

```javascript
// Restore từ backup
db.interactions.drop();
db.interactions_backup.aggregate([
  { $out: "interactions" }
]);

// Verify
db.interactions.countDocuments();
```

## Kết Luận

- **Migration không bắt buộc** - Hệ thống hoạt động bình thường mà không cần migration
- **Nếu muốn clean data** - Chọn Phương Án 2 hoặc 3
- **Luôn backup trước** - Đặc biệt quan trọng trong production
- **Test thoroughly** - Kiểm tra kỹ sau khi migration

## Hỗ Trợ

Nếu cần hỗ trợ thêm, tham khảo:
- Backend code: `post-service/src/main/java/com/ctuconnect/entity/InteractionEntity.java`
- Frontend code: `client-frontend/src/components/ui/ReactionPicker.tsx`
- Main documentation: `REACTION_SYSTEM_UPDATE.md`
