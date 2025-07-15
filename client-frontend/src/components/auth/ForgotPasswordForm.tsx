import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Card from '@/components/ui/Card';
import Link from 'next/link';
import Image from 'next/image';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';
import { authService } from '@/services/authService';
import { useRecaptcha, RECAPTCHA_ACTIONS } from '@/hooks/useRecaptcha';

const forgotPasswordSchema = yup.object({
  email: yup
      .string()
      .required('Email CTU là bắt buộc')
      .email('Email không hợp lệ')
      .test('ctu-email', 'Email phải theo định dạng @ctu.edu.vn hoặc @student.ctu.edu.vn', function(value) {
        if (!value) return false;
        const normalizedValue = value.toLowerCase().trim();
        return normalizedValue.endsWith('@ctu.edu.vn') || normalizedValue.endsWith('@student.ctu.edu.vn');
      }),
});

type ForgotPasswordFormData = yup.InferType<typeof forgotPasswordSchema>;

const ForgotPasswordForm: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const { executeRecaptcha, isReady } = useRecaptcha();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<ForgotPasswordFormData>({
    resolver: yupResolver(forgotPasswordSchema),
  });

  const onSubmit = async (data: ForgotPasswordFormData) => {
    setIsLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Execute reCAPTCHA
      const recaptchaToken = await executeRecaptcha(RECAPTCHA_ACTIONS.FORGOT_PASSWORD);

      if (!recaptchaToken) {
        setError('Xác thực bảo mật thất bại. Vui lòng thử lại.');
        return;
      }

      await authService.forgotPassword(data.email);
      setSuccess('Email hướng dẫn đặt lại mật khẩu đã được gửi đến hộp thư của bạn.');
    } catch (err: unknown) {
      const error = err as { response?: { data?: { message?: string } } };
      setError(error.response?.data?.message || 'Có lỗi xảy ra, vui lòng thử lại.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <div className="flex justify-center">
            <Image
              className="h-12 w-auto"
              src="/logo.svg"
              alt="CTU Connect"
              width={48}
              height={48}
            />
          </div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Quên mật khẩu
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Nhập email của bạn để nhận hướng dẫn đặt lại mật khẩu
          </p>
        </div>

        <Card className="mt-8 space-y-6">
          <div className="rounded-md shadow-sm">
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-md p-4">
                  <div className="flex">
                    <div className="ml-3">
                      <p className="text-sm text-red-800">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {success && (
                <div className="bg-green-50 border border-green-200 rounded-md p-4">
                  <div className="flex">
                    <div className="ml-3">
                      <p className="text-sm text-green-800">{success}</p>
                    </div>
                  </div>
                </div>
              )}

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                  Email CTU <span className="text-red-500">*</span>
                </label>
                <div className="mt-1">
                  <Input
                    id="email"
                    type="email"
                    autoComplete="email"
                    required
                    placeholder="example@ctu.edu.vn"
                    {...register('email')}
                    className={errors.email ? 'border-red-300' : ''}
                  />
                  {errors.email && (
                    <p className="mt-2 text-sm text-red-600">{errors.email.message}</p>
                  )}
                  <p className="mt-1 text-xs text-gray-500">
                    Chỉ chấp nhận email chính thức của trường Đại học Cần Thơ
                  </p>
                </div>
              </div>

              <div>
                <Button
                  type="submit"
                  className="w-full"
                  disabled={isLoading || !isReady}
                >
                  {isLoading ? 'Đang gửi...' : 'Gửi hướng dẫn'}
                </Button>
              </div>

              {/* reCAPTCHA Info */}
              <div className="flex items-center justify-center space-x-2 text-xs text-gray-500">
                <ShieldCheckIcon className="h-4 w-4" />
                <span>Được bảo vệ bởi reCAPTCHA</span>
              </div>

              <div className="text-center">
                <Link href="/login" className="text-sm text-indigo-600 hover:text-indigo-500">
                  Quay lại đăng nhập
                </Link>
              </div>
            </form>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default ForgotPasswordForm;
