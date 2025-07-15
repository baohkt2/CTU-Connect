'use client';

import React, { useEffect, useState, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { authService } from '@/services/authService';
import Button from '@/components/ui/Button';
import Card from '@/components/ui/Card';
import ErrorAlert from '@/components/ui/ErrorAlert';
import Loading from '@/components/ui/Loading';
import Link from 'next/link';
import Image from 'next/image';
import { CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';
import { AxiosError } from 'axios';

type VerifyStatus = 'loading' | 'success' | 'error' | 'expired';

const VerifyEmailContent: React.FC = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [status, setStatus] = useState<VerifyStatus>('loading');
  const [message, setMessage] = useState('');
  const [isResendingEmail, setIsResendingEmail] = useState(false);
  const [resendMessage, setResendMessage] = useState('');

  const token = searchParams.get('token');

  useEffect(() => {
    console.log('Token from URL:', token);
    if (!token) {
      setStatus('error');
      setMessage('Token xác thực không hợp lệ hoặc thiếu.');
      return;
    }
    verifyEmail(token);
  }, [token]);

  const verifyEmail = async (verificationToken: string) => {
    console.log('Verifying email with token:', verificationToken);
    try {
      const response = await authService.verifyEmail(verificationToken);
      console.log('Verification response:', response);
      setStatus('success');
      setMessage('Email đã được xác thực thành công!');
    } catch (err: unknown) {
      const error = err as AxiosError<{ message?: string }>;
      console.error('Email verification failed:', error);

      const errorMessage = error.response?.data?.message || 'Token xác thực không hợp lệ.';

      if (error.response?.status === 400) {
        if (errorMessage.toLowerCase().includes('expired') || errorMessage.includes('hết hạn')) {
          setStatus('expired');
          setMessage('Link xác thực đã hết hạn. Vui lòng yêu cầu gửi lại email xác thực.');
        } else {
          setStatus('error');
          setMessage(errorMessage);
        }
      } else {
        setStatus('error');
        setMessage('Đã xảy ra lỗi khi xác thực email. Vui lòng thử lại sau.');
      }
    }
  };

  const handleResendEmail = async () => {
    if (!token) return;
    setIsResendingEmail(true);
    setResendMessage('');

    try {
      await authService.resendVerificationEmail(token);
      setResendMessage('Email xác thực đã được gửi lại. Vui lòng kiểm tra hộp thư.');
    } catch (err: unknown) {
      console.error('Resend verification email failed:', err);
      setResendMessage('Không thể gửi lại email xác thực. Vui lòng thử lại sau.');
    } finally {
      setIsResendingEmail(false);
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'loading':
        return <Loading size="large" text="Đang xác thực email..." />;
      case 'success':
        return <CheckCircleIcon className="h-16 w-16 text-green-500" />;
      case 'error':
      case 'expired':
        return <XCircleIcon className="h-16 w-16 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'loading':
        return 'text-blue-600';
      case 'success':
        return 'text-green-600';
      case 'error':
      case 'expired':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getTitle = () => {
    switch (status) {
      case 'loading':
        return 'Đang xác thực email...';
      case 'success':
        return 'Xác thực thành công!';
      case 'error':
        return 'Xác thực thất bại';
      case 'expired':
        return 'Link đã hết hạn';
      default:
        return 'Xác thực email';
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 via-white to-cyan-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <div className="flex justify-center mb-6">
            <Image
              className="h-12 w-auto"
              src="/logo.svg"
              alt="CTU Connect"
              width={48}
              height={48}
            />
          </div>
          <div className="flex justify-center mb-6">{getStatusIcon()}</div>
          <h2 className={`text-center text-3xl font-extrabold ${getStatusColor()}`}>
            {getTitle()}
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">{message}</p>
        </div>

        <Card className="mt-8 space-y-6">
          <div className="text-center space-y-4">
            {status === 'loading' && (
              <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                <p className="text-sm text-blue-800">
                  Đang xác thực email của bạn, vui lòng chờ...
                </p>
              </div>
            )}

            {status === 'success' && (
              <>
                <div className="bg-green-50 border border-green-200 rounded-md p-4">
                  <p className="text-sm text-green-800">
                    <strong>Chúc mừng!</strong> Email của bạn đã được xác thực thành công.
                    Bạn có thể đăng nhập vào tài khoản ngay bây giờ.
                  </p>
                </div>
                <div className="pt-4 space-y-3">
                  <Button onClick={() => router.push('/login')} className="w-full">
                    Đăng nhập ngay
                  </Button>
                  <Button onClick={() => router.push('/')} variant="outline" className="w-full">
                    Về trang chủ
                  </Button>
                </div>
              </>
            )}

            {status === 'expired' && (
              <>
                <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                  <p className="text-sm text-yellow-800">
                    <strong>Link đã hết hạn:</strong> Link xác thực chỉ có hiệu lực trong 24 giờ.
                    Vui lòng yêu cầu gửi lại email xác thực.
                  </p>
                </div>

                {resendMessage && (
                  <ErrorAlert
                    message={resendMessage}
                    type={resendMessage.includes('đã được gửi') ? 'info' : 'error'}
                    onClose={() => setResendMessage('')}
                  />
                )}

                <div className="pt-4 space-y-3">
                  <Button
                    onClick={handleResendEmail}
                    disabled={isResendingEmail}
                    loading={isResendingEmail}
                    className="w-full"
                  >
                    Gửi lại email xác thực
                  </Button>
                  <Button
                    onClick={() => router.push('/register')}
                    variant="outline"
                    className="w-full"
                  >
                    Đăng ký lại
                  </Button>
                </div>
              </>
            )}

            {status === 'error' && (
              <>
                <div className="bg-red-50 border border-red-200 rounded-md p-4">
                  <p className="text-sm text-red-800">
                    <strong>Lỗi xác thực:</strong> {message}
                  </p>
                </div>
                <div className="pt-4 space-y-3">
                  <Button onClick={() => router.push('/register')} className="w-full">
                    Đăng ký lại
                  </Button>
                  <Button
                    onClick={() => router.push('/login')}
                    variant="outline"
                    className="w-full"
                  >
                    Đăng nhập
                  </Button>
                </div>
              </>
            )}

            <div className="text-center pt-4">
              <p className="text-xs text-gray-500">
                Cần hỗ trợ?{' '}
                <Link href="/contact" className="text-indigo-600 hover:text-indigo-500">
                  Liên hệ hỗ trợ
                </Link>
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

const VerifyEmailPage: React.FC = () => {
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center"><Loading size="large" text="Đang tải..." /></div>}>
      <VerifyEmailContent />
    </Suspense>
  );
};

export default VerifyEmailPage;
