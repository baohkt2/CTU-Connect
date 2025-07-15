import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Card from '@/components/ui/Card';
import ErrorAlert from '@/components/ui/ErrorAlert';
import Link from 'next/link';
import Image from 'next/image';
import { EyeIcon, EyeSlashIcon, CheckCircleIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';
import { authService } from '@/services/authService';
import { useRecaptcha, RECAPTCHA_ACTIONS } from '@/hooks/useRecaptcha';

const registerSchema = yup.object({
  email: yup
    .string()
    .required('Email CTU là bắt buộc')
    .email('Email không hợp lệ')
    .test('ctu-email', 'Email phải theo định dạng @ctu.edu.vn hoặc @student.ctu.edu.vn', function(value) {
      if (!value) return false;
      const normalizedValue = value.toLowerCase().trim();
      return normalizedValue.endsWith('@ctu.edu.vn') || normalizedValue.endsWith('@student.ctu.edu.vn');
    }),
  username: yup
    .string()
    .required('Tên đăng nhập là bắt buộc')
    .min(3, 'Tên đăng nhập phải có ít nhất 3 ký tự')
    .max(25, 'Tên đăng nhập không được quá 25 ký tự')
    .matches(/^[a-zA-Z][a-zA-Z0-9._]{2,24}$/, 'Tên đăng nhập phải bắt đầu bằng chữ cái và chỉ chứa chữ cái, số, dấu chấm và gạch dưới'),
  password: yup
    .string()
    .min(8, 'Mật khẩu phải có ít nhất 8 ký tự')
    .max(20, 'Mật khẩu không được quá 20 ký tự')
    .matches(
      /^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=!])(?!.*\s).{8,20}$/,
      'Mật khẩu phải chứa ít nhất: 1 chữ số, 1 chữ thường, 1 chữ hoa, 1 ký tự đặc biệt và không có khoảng trắng'
    )
    .required('Mật khẩu là bắt buộc'),
  confirmPassword: yup
    .string()
    .oneOf([yup.ref('password')], 'Mật khẩu xác nhận không khớp')
    .required('Xác nhận mật khẩu là bắt buộc'),
});

type RegisterFormData = yup.InferType<typeof registerSchema>;

const RegisterForm: React.FC = () => {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorType, setErrorType] = useState<'error' | 'warning' | 'info'>('error');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [registrationSuccess, setRegistrationSuccess] = useState(false);
  const { executeRecaptcha, isReady } = useRecaptcha();

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<RegisterFormData>({
    resolver: yupResolver(registerSchema),
  });

  const password = watch('password');

  const onSubmit = async (data: RegisterFormData) => {
    setIsLoading(true);
    setError(null);

    // @ts-ignore
    try {
      // Execute reCAPTCHA
      const recaptchaToken = await executeRecaptcha(RECAPTCHA_ACTIONS.REGISTER);

      if (!recaptchaToken) {
        setError('Xác thực bảo mật thất bại. Vui lòng thử lại.');
        setErrorType('warning');
        return;
      }

      // Normalize email and username to lowercase before sending to backend
      const registerData = {
        email: data.email.toLowerCase().trim(),
        username: data.username.toLowerCase().trim(),
        password: data.password,
        recaptchaToken,
      };

      const respone = await authService.register(registerData);
      console.log('Registration response:', respone);
      setRegistrationSuccess(true);
    } catch (err: any) {
      console.error('Registration error:', err);

      // Xử lý các lỗi cụ thể
      if (err.response?.data?.errorCode) {
        switch (err.response.data.errorCode) {
          case 'EMAIL_ALREADY_EXISTS':
            setError('Email này đã được đăng ký. Vui lòng sử dụng email khác hoặc đăng nhập.');
            setErrorType('warning');
            break;
          case 'USERNAME_ALREADY_EXISTS':
            setError('Tên đăng nhập này đã được sử dụng. Vui lòng chọn tên khác.');
            setErrorType('warning');
            break;
          default:
            setError(err.response.data.message || 'Đăng ký thất bại. Vui lòng thử lại.');
            setErrorType('error');
        }
      } else {
        setError(err.response?.data?.message || 'Đăng ký thất bại. Vui lòng thử lại.');
        setErrorType('error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const getPasswordStrength = () => {
    if (!password) return 0;
    let strength = 0;
    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/\d/.test(password)) strength++;
    if (/[@$!%*?&]/.test(password)) strength++;
    return strength;
  };

  const getPasswordStrengthText = () => {
    const strength = getPasswordStrength();
    if (strength < 2) return { text: 'Yếu', color: 'text-red-600' };
    if (strength < 4) return { text: 'Trung bình', color: 'text-yellow-600' };
    return { text: 'Mạnh', color: 'text-green-600' };
  };

  // Hiển thị trang thành công khi đăng ký thành công
  if (registrationSuccess) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 via-white to-cyan-50 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            <div className="flex justify-center">
              <CheckCircleIcon className="h-16 w-16 text-green-500" />
            </div>
            <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
              Đăng ký thành công!
            </h2>
            <p className="mt-2 text-center text-sm text-gray-600">
              Một email xác thực đã được gửi đến địa chỉ email c���a bạn.
            </p>
          </div>

          <Card className="mt-8 space-y-6">
            <div className="text-center space-y-4">
              <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                <p className="text-sm text-blue-800">
                  <strong>Vui lòng kiểm tra email</strong> và nhấp vào liên kết xác thực để kích hoạt tài khoản của bạn.
                </p>
              </div>

              <div className="text-sm text-gray-600">
                <p>Không nhận được email?</p>
                <p>• Kiểm tra thư mục spam/junk</p>
                <p>• Đảm bảo email đúng định dạng @ctu.edu.vn</p>
              </div>

              <div className="pt-4">
                <Button
                  onClick={() => router.push('/login')}
                  className="w-full"
                >
                  Đến trang đăng nhập
                </Button>
              </div>

              <div className="text-center">
                <button
                  onClick={() => setRegistrationSuccess(false)}
                  className="text-sm text-indigo-600 hover:text-indigo-500"
                >
                  Đăng ký lại với email khác
                </button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 via-white to-cyan-50 py-12 px-4 sm:px-6 lg:px-8">
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
            Tạo tài khoản CTU Connect
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Hoặc{' '}
            <Link href="/login" className="font-medium text-indigo-600 hover:text-indigo-500">
              đăng nhập vào tài khoản có sẵn
            </Link>
          </p>
        </div>

        <Card className="mt-8 space-y-6">
          <div className="rounded-md shadow-sm">
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              {error && (
                <ErrorAlert
                  message={error}
                  type={errorType}
                  onClose={() => setError(null)}
                />
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
                    Sử dụng email chính thức của trường Đại học Cần Thơ
                  </p>
                </div>
              </div>

              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                  Tên đăng nhập <span className="text-red-500">*</span>
                </label>
                <div className="mt-1">
                  <Input
                    id="username"
                    type="text"
                    autoComplete="username"
                    required
                    placeholder="username123"
                    {...register('username')}
                    className={errors.username ? 'border-red-300' : ''}
                  />
                  {errors.username && (
                    <p className="mt-2 text-sm text-red-600">{errors.username.message}</p>
                  )}
                  <p className="mt-1 text-xs text-gray-500">
                    3-30 ký tự, chỉ chứa chữ cái, số và dấu gạch dưới
                  </p>
                </div>
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                  Mật khẩu <span className="text-red-500">*</span>
                </label>
                <div className="mt-1 relative">
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    placeholder="Nhập mật khẩu mạnh"
                    {...register('password')}
                    className={errors.password ? 'border-red-300' : ''}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeSlashIcon className="h-5 w-5 text-gray-400" />
                    ) : (
                      <EyeIcon className="h-5 w-5 text-gray-400" />
                    )}
                  </button>
                </div>
                {password && (
                  <div className="mt-2">
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-300 ${
                            getPasswordStrength() < 2 ? 'bg-red-500' :
                            getPasswordStrength() < 4 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${(getPasswordStrength() / 5) * 100}%` }}
                        />
                      </div>
                      <span className={`text-xs font-medium ${getPasswordStrengthText().color}`}>
                        {getPasswordStrengthText().text}
                      </span>
                    </div>
                  </div>
                )}
                {errors.password && (
                  <p className="mt-2 text-sm text-red-600">{errors.password.message}</p>
                )}
                <p className="mt-1 text-xs text-gray-500">
                  Ít nhất 8 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt
                </p>
              </div>

              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                  Xác nhận mật khẩu <span className="text-red-500">*</span>
                </label>
                <div className="mt-1 relative">
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    placeholder="Nhập lại mật khẩu"
                    {...register('confirmPassword')}
                    className={errors.confirmPassword ? 'border-red-300' : ''}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? (
                      <EyeSlashIcon className="h-5 w-5 text-gray-400" />
                    ) : (
                      <EyeIcon className="h-5 w-5 text-gray-400" />
                    )}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <p className="mt-2 text-sm text-red-600">{errors.confirmPassword.message}</p>
                )}
              </div>

              <div>
                <Button
                  type="submit"
                  className="w-full"
                  disabled={isLoading || !isReady}
                  loading={isLoading}
                >
                  Tạo tài khoản
                </Button>
              </div>

              {/* reCAPTCHA Info */}
              <div className="flex items-center justify-center space-x-2 text-xs text-gray-500">
                <ShieldCheckIcon className="h-4 w-4" />
                <span>Được bảo vệ bởi reCAPTCHA</span>
              </div>

              <div className="text-center">
                <p className="text-xs text-gray-500">
                  Sau khi đăng ký, bạn sẽ nhận được email xác thực.{' '}
                  <br />
                  Vui lòng kiểm tra email và xác thực để hoàn tất đăng ký.
                </p>
              </div>
            </form>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default RegisterForm;
