package vn.ctu.edu.recommend.model.enums;

/**
 * Graph relationship types between users
 */
public enum RelationshipType {
    FRIEND,             // Direct friendship
    SAME_MAJOR,         // Same major/specialization
    SAME_FACULTY,       // Same faculty/department
    SAME_BATCH,         // Same batch/cohort
    SAME_CLASS,         // Same class
    FOLLOWS,            // User follows another
    POSTED,             // User posted content
    INTERACTED_WITH     // User interacted with post
}
