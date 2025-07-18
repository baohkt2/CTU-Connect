import React, {useState} from 'react';
import {useRouter, useSearchParams} from 'next/navigation';
import {useForm} from 'react-hook-form';
import {yupResolver} from '@hookform/resolvers/yup';
import * as yup from 'yup';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Card from '@/components/ui/Card';
import Link from 'next/link';
import Image from 'next/image';
import {EyeIcon, EyeSlashIcon} from '@heroicons/react/24/outline';
import {authService} from '@/services/authService';

const resetPasswordSchema = yup.object({
    password: yup.string().min(6, 'Mật khẩu phải có ít nhất 6 ký tự').required('Mật khẩu là bắt buộc'),
    confirmPassword: yup.string().oneOf([yup.ref('password')], 'Mật khẩu xác nhận không khớp').required('Xác nhận mật khẩu là bắt buộc'),
});

type ResetPasswordFormData = yup.InferType<typeof resetPasswordSchema>;

const ResetPasswordForm: React.FC = () => {
    const router = useRouter();
    const searchParams = useSearchParams();
    const token = searchParams.get('token');

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);

    const {
        register,
        handleSubmit,
        formState: {errors},
        watch
    } = useForm<ResetPasswordFormData>({
        resolver: yupResolver(resetPasswordSchema),
    });
    const password = watch('password');
    const onSubmit = async (data: ResetPasswordFormData) => {
        if (!token) {
            setError('Token không hợp lệ');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            await authService.resetPassword(token, data.password);
            router.push('/login?message=password-reset-success');
        } catch (err: unknown) {
            const error = err as { response?: { data?: { message?: string } } };
            setError(error.response?.data?.message || 'Có lỗi xảy ra, vui lòng thử lại.');
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
        if (strength < 2) return {text: 'Yếu', color: 'text-red-600'};
        if (strength < 4) return {text: 'Trung bình', color: 'text-yellow-600'};
        return {text: 'Mạnh', color: 'text-green-600'};
    };
    if (!token) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
                <div className="max-w-md w-full space-y-8">
                    <div className="text-center">
                        <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
                            Token không hợp lệ
                        </h2>
                        <p className="mt-2 text-center text-sm text-gray-600">
                            Vui lòng kiểm tra lại liên kết trong email của bạn.
                        </p>
                        <div className="mt-4">
                            <Link href="/login" className="text-indigo-600 hover:text-indigo-500">
                                Quay lại đăng nhập
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

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
                        Đặt lại mật khẩu
                    </h2>
                    <p className="mt-2 text-center text-sm text-gray-600">
                        Nhập mật khẩu mới cho tài khoản của bạn
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

                            <div>
                                <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                                    Mật khẩu mới
                                </label>
                                <div className="mt-1 relative">
                                    <Input
                                        id="password"
                                        type={showPassword ? 'text' : 'password'}
                                        autoComplete="new-password"
                                        required
                                        {...register('password')}
                                        className={errors.password ? 'border-red-300' : ''}
                                    />
                                    <button
                                        type="button"
                                        className="absolute inset-y-0 right-0 pr-3 flex items-center"
                                        onClick={() => setShowPassword(!showPassword)}
                                    >
                                        {showPassword ? (
                                            <EyeSlashIcon className="h-5 w-5 text-gray-400"/>
                                        ) : (
                                            <EyeIcon className="h-5 w-5 text-gray-400"/>
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
                                                    style={{width: `${(getPasswordStrength() / 5) * 100}%`}}
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
                            </div>

                            <div>
                                <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                                    Xác nhận mật khẩu mới
                                </label>
                                <div className="mt-1 relative">
                                    <Input
                                        id="confirmPassword"
                                        type={showConfirmPassword ? 'text' : 'password'}
                                        autoComplete="new-password"
                                        required
                                        {...register('confirmPassword')}
                                        className={errors.confirmPassword ? 'border-red-300' : ''}
                                    />
                                    <button
                                        type="button"
                                        className="absolute inset-y-0 right-0 pr-3 flex items-center"
                                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                    >
                                        {showConfirmPassword ? (
                                            <EyeSlashIcon className="h-5 w-5 text-gray-400"/>
                                        ) : (
                                            <EyeIcon className="h-5 w-5 text-gray-400"/>
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
                                    disabled={isLoading}
                                >
                                    {isLoading ? 'Đang đặt lại...' : 'Đặt lại mật khẩu'}
                                </Button>
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

export default ResetPasswordForm;
