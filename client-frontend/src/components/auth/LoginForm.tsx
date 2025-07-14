import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Card from '@/components/ui/Card';
import Link from 'next/link';
import Image from 'next/image';
import { EyeIcon, EyeSlashIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';
import { useRecaptcha, RECAPTCHA_ACTIONS } from '@/hooks/useRecaptcha';
import { AxiosError } from 'axios';

// Define the error response structure
interface ApiErrorResponse {
  message: string;
}

// Define the LoginFormData interface (inferred from Yup schema)
interface LoginFormData {
  emailOrUsername: string;
  password: string;
}

// Yup schema for form validation
const loginSchema = yup.object({
  emailOrUsername: yup
      .string()
      .required('Email hoặc tên đăng nhập là bắt buộc')
      .test(
          'valid-input',
          'Email phải kết thúc bằng @ctu.edu.vn hoặc là tên đăng nhập hợp lệ',
          function (value) {
            if (!value) return false;
            if (value.includes('@')) {
              return /^[A-Za-z0-9._%+-]+@(student\.)?ctu\.edu\.vn$/.test(value);
            }
            return /^[a-zA-Z][a-zA-Z0-9._]{2,24}$/.test(value);
          }
      ),
  password: yup
      .string()
      .required('Mật khẩu là bắt buộc')
      .min(8, 'Mật khẩu phải có ít nhất 8 ký tự')
      .max(20, 'Mật khẩu không được vượt quá 20 ký tự')
      .matches(
          /^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=!])(?!.*\s).{8,20}$/,
          'Mật khẩu cần chứa chữ hoa, chữ thường, số, ký tự đặc biệt và không có khoảng trắng'
      ),
});

const LoginForm: React.FC = () => {
  const { login } = useAuth();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const { executeRecaptcha, isReady } = useRecaptcha();

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<LoginFormData>({
    resolver: yupResolver(loginSchema),
  });

  const emailOrUsername = watch('emailOrUsername');

  const onSubmit = async (data: LoginFormData) => {
    setIsLoading(true);
    setError(null);

    try {
      // Execute reCAPTCHA
      const recaptchaToken = await executeRecaptcha(RECAPTCHA_ACTIONS.LOGIN);

      if (!recaptchaToken) {
        setError('Xác thực bảo mật thất bại. Vui lòng thử lại.');
        return;
      }

      // Chuyển đổi data để phù hợp với backend API
      const loginData = data.emailOrUsername.includes('@')
        ? { email: data.emailOrUsername, password: data.password, recaptchaToken }
        : { username: data.emailOrUsername, password: data.password, recaptchaToken };

      await login(loginData);
      router.push('/');
    } catch (err: unknown) {
      if (err instanceof AxiosError) {
        const apiError = err as AxiosError<ApiErrorResponse>;
        setError(apiError.response?.data?.message || 'Đăng nhập thất bại. Vui lòng thử lại.');
      } else {
        setError('Đã xảy ra lỗi không xác định. Vui lòng thử lại.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const getInputType = () => (emailOrUsername?.includes('@') ? 'email' : 'text');
  const getInputPlaceholder = () =>
      emailOrUsername?.includes('@') ? 'example@student.ctu.edu.vn' : 'Tên đăng nhập';

  return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-100 via-white to-cyan-100 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            <Image src="/images/logo.png" alt="CTU Connect" width={150} height={150} className="mx-auto" />
            <h2 className="mt-6 text-3xl font-bold text-gray-800">Đăng nhập vào CTU Connect</h2>
            <p className="mt-2 text-sm text-gray-600">
              Chưa có tài khoản?{' '}
              <Link href="/register" className="text-indigo-600 font-medium hover:underline">
                Đăng ký ngay
              </Link>
            </p>
          </div>

          <Card className="p-6 space-y-6">
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
              {error && <div className="text-sm text-red-600 bg-red-100 p-2 rounded">{error}</div>}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Email CTU hoặc Tên đăng nhập
                </label>
                <Input
                    type={getInputType()}
                    placeholder={getInputPlaceholder()}
                    autoComplete="username"
                    {...register('emailOrUsername')}
                    className={errors.emailOrUsername ? 'border-red-500' : ''}
                />
                {errors.emailOrUsername && (
                    <p className="text-sm text-red-600 mt-1">{errors.emailOrUsername.message}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Mật khẩu</label>
                <div className="relative">
                  <Input
                      type={showPassword ? 'text' : 'password'}
                      placeholder="Nhập mật khẩu"
                      autoComplete="current-password"
                      {...register('password')}
                      className={errors.password ? 'border-red-500' : ''}
                  />
                  <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-700"
                  >
                    {showPassword ? <EyeSlashIcon className="h-5 w-5" /> : <EyeIcon className="h-5 w-5" />}
                  </button>
                </div>
                {errors.password && <p className="text-sm text-red-600 mt-1">{errors.password.message}</p>}
              </div>

              <div className="flex justify-end">
                <Link href="/forgot-password" className="text-sm text-indigo-600 hover:underline">
                  Quên mật khẩu?
                </Link>
              </div>

              <div>
                <Button
                  type="submit"
                  className="w-full"
                  disabled={isLoading || !isReady}
                  loading={isLoading}
                >
                  Đăng nhập
                </Button>
              </div>

              {/* reCAPTCHA Info */}
              <div className="flex items-center justify-center space-x-2 text-xs text-gray-500">
                <ShieldCheckIcon className="h-4 w-4" />
                <span>Được bảo vệ bởi reCAPTCHA</span>
              </div>
            </form>
          </Card>
        </div>
      </div>
  );
};

export default LoginForm;

