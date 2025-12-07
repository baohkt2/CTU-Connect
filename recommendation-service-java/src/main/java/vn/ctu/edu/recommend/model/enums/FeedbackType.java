package vn.ctu.edu.recommend.model.enums;

/**
 * User feedback types for reinforcement learning
 */
public enum FeedbackType {
    VIEW,           // User viewed the post
    CLICK,          // User clicked on the post
    LIKE,           // User liked the post
    COMMENT,        // User commented on the post
    SHARE,          // User shared the post
    SAVE,           // User saved the post
    SKIP,           // User skipped the post
    HIDE,           // User hid the post
    REPORT,         // User reported the post
    DWELL_TIME      // Time user spent on post
}
