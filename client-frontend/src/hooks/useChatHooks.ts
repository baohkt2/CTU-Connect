import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { chatService } from '@/services/chatService';
import type {
  ConversationResponse,
  MessageResponse,
  CreateConversationRequest,
  SendMessageRequest,
  UpdateConversationRequest,
  AddReactionRequest
} from '@/services/chatService';

export const useChatHooks = () => {
  const queryClient = useQueryClient();

  // Get chat conversations (updated method name)
  const useChatConversations = (page = 0, size = 20) => {
    return useQuery({
      queryKey: ['chat', 'conversations', page, size],
      queryFn: () => chatService.getUserConversations(page, size),
      refetchInterval: 5000, // Refetch every 5 seconds
    });
  };

  // Get single conversation (updated method name)
  const useConversation = (conversationId: string) => {
    return useQuery({
      queryKey: ['chat', 'conversation', conversationId],
      queryFn: () => chatService.getConversation(conversationId),
      enabled: !!conversationId,
    });
  };

  // Get messages for a conversation
  const useMessages = (conversationId: string, page = 0, size = 50) => {
    return useQuery({
      queryKey: ['chat', 'messages', conversationId, page, size],
      queryFn: () => chatService.getMessages(conversationId, page, size),
      enabled: !!conversationId,
      refetchInterval: 2000, // Refetch every 2 seconds
    });
  };

  // Get unread count
  const useUnreadCount = () => {
    return useQuery({
      queryKey: ['chat', 'unread'],
      queryFn: () => chatService.getUnreadCount(),
      refetchInterval: 10000, // Refetch every 10 seconds
    });
  };

  // Create conversation mutation
  const useCreateConversation = () => {
    return useMutation({
      mutationFn: (request: CreateConversationRequest) => chatService.createConversation(request),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
      },
    });
  };

  // Create direct conversation mutation (helper)
  const useCreateDirectConversation = () => {
    return useMutation({
      mutationFn: (participantId: string) => chatService.createDirectConversation(participantId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
      },
    });
  };

  // Update conversation mutation
  const useUpdateConversation = () => {
    return useMutation({
      mutationFn: ({ conversationId, request }: { conversationId: string; request: UpdateConversationRequest }) =>
        chatService.updateConversation(conversationId, request),
      onSuccess: (data) => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversation', data.id] });
      },
    });
  };

  // Send message mutation
  const useSendMessage = () => {
    return useMutation({
      mutationFn: (request: SendMessageRequest) => chatService.sendMessage(request),
      onSuccess: (data) => {
        // Invalidate messages for the conversation
        queryClient.invalidateQueries({ queryKey: ['chat', 'messages', data.conversationId] });
        // Update conversations to reflect new last message
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
        // Update unread count
        queryClient.invalidateQueries({ queryKey: ['chat', 'unread'] });
      },
    });
  };

  // Edit message mutation
  const useEditMessage = () => {
    return useMutation({
      mutationFn: ({ messageId, content }: { messageId: string; content: string }) =>
        chatService.editMessage(messageId, content),
      onSuccess: (data) => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'messages', data.conversationId] });
      },
    });
  };

  // Delete message mutation
  const useDeleteMessage = () => {
    return useMutation({
      mutationFn: (messageId: string) => chatService.deleteMessage(messageId),
      onSuccess: () => {
        // Invalidate all messages queries since we don't know which conversation
        queryClient.invalidateQueries({ queryKey: ['chat', 'messages'] });
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
      },
    });
  };

  // Add reaction mutation
  const useAddReaction = () => {
    return useMutation({
      mutationFn: (request: AddReactionRequest) => chatService.addReaction(request),
      onSuccess: () => {
        // Invalidate messages to show updated reactions
        queryClient.invalidateQueries({ queryKey: ['chat', 'messages'] });
      },
    });
  };

  // Remove reaction mutation
  const useRemoveReaction = () => {
    return useMutation({
      mutationFn: ({ messageId, emoji }: { messageId: string; emoji: string }) =>
        chatService.removeReaction(messageId, emoji),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'messages'] });
      },
    });
  };

  // Mark as read mutation
  const useMarkAsRead = () => {
    return useMutation({
      mutationFn: (conversationId: string) => chatService.markAsRead(conversationId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
        queryClient.invalidateQueries({ queryKey: ['chat', 'unread'] });
      },
    });
  };

  // Add participant mutation
  const useAddParticipant = () => {
    return useMutation({
      mutationFn: ({ conversationId, participantId }: { conversationId: string; participantId: string }) =>
        chatService.addParticipant(conversationId, participantId),
      onSuccess: (_, variables) => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversation', variables.conversationId] });
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
      },
    });
  };

  // Remove participant mutation
  const useRemoveParticipant = () => {
    return useMutation({
      mutationFn: ({ conversationId, participantId }: { conversationId: string; participantId: string }) =>
        chatService.removeParticipant(conversationId, participantId),
      onSuccess: (_, variables) => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversation', variables.conversationId] });
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
      },
    });
  };

  // Search conversations
  const useSearchConversations = () => {
    return useMutation({
      mutationFn: (query: string) => chatService.searchConversations(query),
    });
  };

  // File upload for chat
  const useUploadChatFile = () => {
    return useMutation({
      mutationFn: (file: File) => chatService.uploadChatFile(file),
    });
  };

  // User presence hooks
  const useUserPresence = (userId: string) => {
    return useQuery({
      queryKey: ['chat', 'presence', userId],
      queryFn: () => chatService.getUserPresence(userId),
      enabled: !!userId,
      refetchInterval: 30000, // Refetch every 30 seconds
    });
  };

  const useOnlineUsers = () => {
    return useQuery({
      queryKey: ['chat', 'presence', 'online'],
      queryFn: () => chatService.getOnlineUsers(),
      refetchInterval: 30000, // Refetch every 30 seconds
    });
  };

  const useUpdatePresenceStatus = () => {
    return useMutation({
      mutationFn: (status: 'ONLINE' | 'OFFLINE' | 'AWAY') => chatService.updatePresenceStatus(status),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'presence'] });
      },
    });
  };

  return {
    // Queries
    useChatConversations,
    useConversation,
    useMessages,
    useUnreadCount,
    useUserPresence,
    useOnlineUsers,

    // Mutations
    useCreateConversation,
    useCreateDirectConversation,
    useUpdateConversation,
    useSendMessage,
    useEditMessage,
    useDeleteMessage,
    useAddReaction,
    useRemoveReaction,
    useMarkAsRead,
    useAddParticipant,
    useRemoveParticipant,
    useSearchConversations,
    useUploadChatFile,
    useUpdatePresenceStatus,
  };
};
