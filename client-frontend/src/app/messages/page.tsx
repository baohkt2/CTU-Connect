'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import Layout from '@/components/layout/Layout';
import { useRouter, useSearchParams } from 'next/navigation';
import ChatSidebar from '@/components/chat/ChatSidebar';
import ChatMessageArea from '@/components/chat/ChatMessageArea';
import api from '@/lib/api';

function MessagesContent() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
  const [selectedConversation, setSelectedConversation] = useState<any>(null);
  const friendUserId = searchParams.get('userId');

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  // Load conversation details when selected
  useEffect(() => {
    if (selectedConversationId) {
      loadConversationDetails(selectedConversationId);
    }
  }, [selectedConversationId]);

  const loadConversationDetails = async (conversationId: string) => {
    try {
      const response = await api.get(`/chats/conversations/${conversationId}`);
      setSelectedConversation(response.data);
    } catch (error) {
      console.error('Error loading conversation details:', error);
    }
  };

  const getConversationInfo = () => {
    if (!selectedConversation) return {};

    if (selectedConversation.type === 'DIRECT') {
      const otherParticipant = selectedConversation.participants?.[0];
      return {
        name: otherParticipant?.fullName || 'Người dùng',
        avatar: otherParticipant?.avatarUrl,
        isOnline: otherParticipant?.isOnline,
      };
    }

    return {
      name: selectedConversation.name || 'Nhóm chat',
      avatar: selectedConversation.metadata?.avatarUrl,
      isOnline: false,
    };
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  const conversationInfo = getConversationInfo();

  return (
    <Layout>
      <div className="h-[calc(100vh-4rem)] flex bg-white rounded-lg shadow-sm overflow-hidden">
        {/* Sidebar - Danh sách conversations */}
        <ChatSidebar
          selectedConversationId={selectedConversationId}
          onSelectConversation={setSelectedConversationId}
          friendUserId={friendUserId}
        />
        
        {/* Main chat area */}
        <ChatMessageArea
          conversationId={selectedConversationId}
          currentUserId={user.id}
          conversationName={conversationInfo.name}
          conversationAvatar={conversationInfo.avatar}
          isOnline={conversationInfo.isOnline}
        />
      </div>
    </Layout>
  );
}

export default function MessagesPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    }>
      <MessagesContent />
    </Suspense>
  );
}
