import { BaseEntity } from './common';
import { User } from './user';

// Chat room entity
export interface ChatRoom extends BaseEntity {
  name?: string;
  type: ChatRoomType;
  participants: User[];
  lastMessage?: ChatMessage;
  unreadCount: number;
  isActive: boolean;
  createdBy: string;
}

// Chat message entity
export interface ChatMessage extends BaseEntity {
  content: string;
  type: MessageType;
  roomId: string;
  senderId: string;
  sender: User;
  isRead: boolean;
  readBy?: MessageReadStatus[];
  replyTo?: string; // Reply to another message
  attachments?: MessageAttachment[];
  isEdited: boolean;
  editedAt?: string;
  reactions: MessageReaction[];
  replyToMessage?: {
    senderName: string;
    content: string;
  };
}

// Message reaction
export interface MessageReaction {
  userId: string;
  emoji: string;
  createdAt: string;
}

// Chat room types
export enum ChatRoomType {
  DIRECT = 'DIRECT',
  GROUP = 'GROUP',
}

// Message types
export enum MessageType {
  TEXT = 'TEXT',
  IMAGE = 'IMAGE',
  FILE = 'FILE',
  SYSTEM = 'SYSTEM',
}

// Message read status
export interface MessageReadStatus {
  userId: string;
  user: User;
  readAt: string;
}

// Message attachment
export interface MessageAttachment {
  id: string;
  filename: string;
  url: string;
  type: string;
  size: number;
}

// Chat operations
export interface CreateChatRoomRequest {
  participantIds: string[];
  name?: string;
  type: ChatRoomType;
}

export interface SendMessageRequest {
  content: string;
  type: MessageType;
  roomId: string;
  replyTo?: string;
  attachments?: File[];
}

export interface UpdateMessageRequest {
  content: string;
}

// Online status
export interface OnlineStatus {
  userId: string;
  isOnline: boolean;
  lastSeen: string;
}
