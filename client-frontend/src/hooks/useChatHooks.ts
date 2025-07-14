import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { chatService } from '@/services/chatService';

export const useChatHooks = () => {
  const queryClient = useQueryClient();

  // Get chat rooms
  const useChatRooms = (page = 0, size = 10) => {
    return useQuery({
      queryKey: ['chat', 'rooms', page, size],
      queryFn: () => chatService.getChatRooms(page, size),
      refetchInterval: 5000, // Refetch every 5 seconds
    });
  };

  // Get single chat room
  const useChatRoom = (roomId: string) => {
    return useQuery({
      queryKey: ['chat', 'room', roomId],
      queryFn: () => chatService.getChatRoom(roomId),
      enabled: !!roomId,
    });
  };

  // Get messages for a chat room
  const useMessages = (roomId: string, page = 0, size = 20) => {
    return useQuery({
      queryKey: ['chat', 'messages', roomId, page, size],
      queryFn: () => chatService.getMessages(roomId, page, size),
      enabled: !!roomId,
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

  // Create chat room mutation
  const useCreateChatRoom = () => {
    return useMutation({
      mutationFn: (participantId: string) => chatService.createChatRoom(participantId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'rooms'] });
      },
    });
  };

  // Send message mutation
  const useSendMessage = () => {
    return useMutation({
      mutationFn: ({ roomId, content }: { roomId: string; content: string }) =>
        chatService.sendMessage(roomId, content),
      onSuccess: (_, { roomId }) => {
        queryClient.invalidateQueries({ queryKey: ['chat', 'messages', roomId] });
        queryClient.invalidateQueries({ queryKey: ['chat', 'rooms'] });
      },
    });
  };

  // Mark as read mutation
  const useMarkAsRead = () => {
    return useMutation({
      mutationFn: (messageId: string) => chatService.markAsRead(messageId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['chat'] });
      },
    });
  };

  // Mark room as read mutation
  const useMarkRoomAsRead = () => {
    return useMutation({
      mutationFn: (roomId: string) => chatService.markRoomAsRead(roomId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['chat'] });
      },
    });
  };

  return {
    useChatRooms,
    useChatRoom,
    useMessages,
    useUnreadCount,
    useCreateChatRoom,
    useSendMessage,
    useMarkAsRead,
    useMarkRoomAsRead,
  };
};
