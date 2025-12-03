// Comment management utility for proper storage and display structure
import { Comment } from '@/types';

export class CommentManager {
  private static readonly MAX_DEPTH = 3;
  private static readonly REPLIES_PER_PAGE = 5;

  /**
   * Normalize comment data from API response
   */
  static normalizeComment(rawComment: any): Comment {
    return {
      id: rawComment.id,
      content: rawComment.content,
      postId: rawComment.postId,
      author: CommentManager.normalizeAuthor(rawComment.author),
      authorId: rawComment.authorId || rawComment.author?.id,
      parentCommentId: rawComment.parentCommentId || rawComment.parentId,
      parentId: rawComment.parentId || rawComment.parentCommentId,
      rootCommentId: rawComment.rootCommentId,
      depth: rawComment.depth || 0,
      replyToAuthor: rawComment.replyToAuthor,
      stats: rawComment.stats,
      likesCount: rawComment.likesCount || rawComment.stats?.likes || 0,
      createdAt: rawComment.createdAt,
      updatedAt: rawComment.updatedAt,
      replies: rawComment.replies ? rawComment.replies.map(reply => CommentManager.normalizeComment(reply)) : [],
      replyCount: rawComment.replyCount || rawComment.stats?.replies || 0,
      isFlattened: rawComment.isFlattened || false,
      hasMoreReplies: false,
      isLoadingReplies: false,
      showReplies: false,
    };
  }

  /**
   * Normalize author data with fallbacks
   */
  static normalizeAuthor(rawAuthor: any): Comment['author'] {
    if (!rawAuthor) {
      return {
        id: 'unknown',
        name: 'Người dùng',
        fullName: 'Người dùng',
        username: 'user',
        avatar: null,
        avatarUrl: null,
        verified: false,
      };
    }

    return {
      id: rawAuthor.id,
      name: rawAuthor.name || rawAuthor.fullName || rawAuthor.username || 'Người dùng',
      fullName: rawAuthor.fullName || rawAuthor.name || 'Người dùng',
      username: rawAuthor.username || 'user',
      avatar: rawAuthor.avatar || rawAuthor.avatarUrl,
      avatarUrl: rawAuthor.avatarUrl || rawAuthor.avatar,
      verified: rawAuthor.verified || rawAuthor.isVerified || false,
    };
  }

  /**
   * Build hierarchical comment tree from flat list
   */
  static buildCommentTree(comments: Comment[]): Comment[] {
    const commentMap = new Map<string, Comment>();
    const rootComments: Comment[] = [];

    // First pass: create map of all comments
    comments.forEach(comment => {
      const normalized = CommentManager.normalizeComment(comment);
      commentMap.set(normalized.id, normalized);
    });

    // Second pass: build tree structure
    comments.forEach(comment => {
      const normalized = commentMap.get(comment.id);
      if (!normalized) return;

      if (!normalized.parentCommentId && !normalized.parentId) {
        // Root comment
        rootComments.push(normalized);
      } else {
        // Reply comment
        const parentId = normalized.parentCommentId || normalized.parentId;
        const parent = commentMap.get(parentId!);

        if (parent && parent.depth! < CommentManager.MAX_DEPTH) {
          // Add as nested reply
          if (!parent.replies) parent.replies = [];
          parent.replies.push(normalized);
          parent.replyCount = (parent.replyCount || 0) + 1;
        } else {
          // Flatten if too deep
          normalized.isFlattened = true;
          normalized.depth = CommentManager.MAX_DEPTH;
          rootComments.push(normalized);
        }
      }
    });

    return rootComments.sort((a, b) =>
      new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
    );
  }

  /**
   * Add a new reply to existing comment structure
   */
  static addReplyToComment(
    comments: Comment[],
    newReply: Comment,
    parentId: string
  ): Comment[] {
    const normalizedReply = CommentManager.normalizeComment(newReply);

    const addReplyRecursive = (commentList: Comment[]): Comment[] => {
      return commentList.map(comment => {
        if (comment.id === parentId) {
          // Found parent - add reply
          const updatedComment = { ...comment };
          if (!updatedComment.replies) updatedComment.replies = [];

          updatedComment.replies.push(normalizedReply);
          updatedComment.replyCount = (updatedComment.replyCount || 0) + 1;
          updatedComment.showReplies = true; // Auto-show replies when new one is added

          return updatedComment;
        } else if (comment.replies && comment.replies.length > 0) {
          // Recursively search in replies
          return {
            ...comment,
            replies: addReplyRecursive(comment.replies)
          };
        }
        return comment;
      });
    };

    return addReplyRecursive(comments);
  }

  /**
   * Remove a comment from the structure
   */
  static removeComment(comments: Comment[], commentId: string): Comment[] {
    const removeRecursive = (commentList: Comment[]): Comment[] => {
      return commentList
        .filter(comment => comment.id !== commentId)
        .map(comment => ({
          ...comment,
          replies: comment.replies ? removeRecursive(comment.replies) : []
        }));
    };

    return removeRecursive(comments);
  }

  /**
   * Update comment likes count
   */
  static updateCommentLikes(
    comments: Comment[],
    commentId: string,
    newLikesCount: number
  ): Comment[] {
    const updateRecursive = (commentList: Comment[]): Comment[] => {
      return commentList.map(comment => {
        if (comment.id === commentId) {
          return {
            ...comment,
            likesCount: newLikesCount,
            stats: comment.stats ? { ...comment.stats, likes: newLikesCount } : { likes: newLikesCount, replies: 0 }
          };
        } else if (comment.replies && comment.replies.length > 0) {
          return {
            ...comment,
            replies: updateRecursive(comment.replies)
          };
        }
        return comment;
      });
    };

    return updateRecursive(comments);
  }

  /**
   * Toggle replies visibility
   */
  static toggleRepliesVisibility(
    comments: Comment[],
    commentId: string,
    showReplies: boolean
  ): Comment[] {
    const toggleRecursive = (commentList: Comment[]): Comment[] => {
      return commentList.map(comment => {
        if (comment.id === commentId) {
          return { ...comment, showReplies };
        } else if (comment.replies && comment.replies.length > 0) {
          return {
            ...comment,
            replies: toggleRecursive(comment.replies)
          };
        }
        return comment;
      });
    };

    return toggleRecursive(comments);
  }

  /**
   * Get display name for comment author
   */
  static getAuthorDisplayName(author: Comment['author']): string {
    return author.fullName || author.name || author.username || 'Người dùng';
  }

  /**
   * Get display avatar for comment author
   */
  static getAuthorDisplayAvatar(author: Comment['author']): string {
    return author.avatarUrl || author.avatar || '/default-avatar.png';
  }

  /**
   * Check if comment can have replies
   */
  static canAddReply(comment: Comment): boolean {
    return (comment.depth || 0) < CommentManager.MAX_DEPTH;
  }

  /**
   * Calculate total replies count including nested replies
   */
  static getTotalRepliesCount(comment: Comment): number {
    if (!comment.replies || comment.replies.length === 0) {
      return comment.replyCount || 0;
    }

    const directReplies = comment.replies.length;
    const nestedReplies = comment.replies.reduce(
      (total, reply) => total + CommentManager.getTotalRepliesCount(reply),
      0
    );

    return directReplies + nestedReplies;
  }
}
