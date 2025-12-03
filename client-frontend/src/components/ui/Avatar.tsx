import React from 'react';
import {cn} from '@/utils/helpers';
import {useAuth} from '@/contexts/AuthContext';

interface AvatarProps {
    src?: string,
    alt?: string,
    size?: 'sm' | 'md' | 'lg' | 'xl',
    fallback?: string,
    className?: string,
    online?: boolean,
    id?: string,
}

const Avatar: React.FC<AvatarProps> = ({
                                           src,
                                           id,
                                           alt = 'Avatar',
                                           size = 'md',
                                           fallback,
                                           className,
                                           online,

                                       }) => {
    const {user} = useAuth();

    const sizes = {
        sm: 'w-8 h-8 text-sm',
        md: 'w-10 h-10 text-base',
        lg: 'w-12 h-12 text-lg',
        xl: 'w-16 h-16 text-xl',
    };

    const onlineSizes = {
        sm: 'w-2 h-2',
        md: 'w-3 h-3',
        lg: 'w-3 h-3',
        xl: 'w-4 h-4',
    };

    // Hàm xử lý sự kiện click
    const handleClick = () => {
        console.log('Avatar clicked:', id);
        console.log("Avatar current user:", id === user?.id);
        // Nếu có id, chuyển hướng đến trang cá nhân của người dùng
        if (id === user?.id) {
            window.location.replace("/profile/me");
        } else if (id) {
            window.location.replace("/profile/${id}");
        } else {
            // Nếu không có id, có thể hiển thị thông báo hoặc thực hiện hành động khác
            console.warn('No user ID provided for avatar click.');
        }
    }

    return (
        <div className={cn('relative inline-block', className)} onClick={handleClick}>
            <div
                className={cn(
                    'rounded-full bg-gray-200 flex items-center justify-center overflow-hidden',
                    sizes[size],
                    // Thêm lớp hover bóng tròn
                    'transition-shadow duration-300 ease-in-out',
                    'hover:shadow-[0_0_15px_5px_rgba(0,0,0,0.25)]',
                    'hover:rounded-full' // chắc chắn vẫn giữ bo tròn khi hover
                )}
            >
                {src != "/default-avatar.png" ? (
                    <img
                        src={src}
                        alt={alt}
                        className="w-full h-full object-cover"
                    />
                ) : (
                    <span className="font-semibold text-gray-600">
            {fallback || alt.charAt(0).toUpperCase()}
          </span>
                )}
            </div>
            {online && (
                <span
                    className={cn(
                        'absolute bottom-0 right-0 bg-green-500 rounded-full border-2 border-white',
                        onlineSizes[size]
                    )}
                />
            )}
        </div>
    )
}

export default Avatar;
