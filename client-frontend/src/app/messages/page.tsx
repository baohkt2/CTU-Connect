/*
'use client';

import React, { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { ChatProvider } from '@/contexts/ChatContext';
import Layout from '@/components/layout/Layout';
import ChatWindow from '@/components/chat/ChatWindow';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function MessagesPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [isChatOpen, setIsChatOpen] = useState(true);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

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

  return (
    <Layout>
      <div className="h-[calc(100vh-8rem)] max-w-7xl mx-auto">
        <ChatProvider>
          <ChatWindow
            isOpen={isChatOpen}
            onClose={() => setIsChatOpen(false)}
            currentUserId={user.id}
          />
        </ChatProvider>
      </div>
    </Layout>
  );
}
*/
