import React from 'react';
import {
    ExclamationTriangleIcon,
    InformationCircleIcon,
    XCircleIcon,
    XMarkIcon
} from '@heroicons/react/24/outline';

interface ErrorAlertProps {
    message: string;
    type?: 'error' | 'warning' | 'info';
    onClose?: () => void;
    showIcon?: boolean;
    className?: string;
}

const alertStyles = {
    error: 'bg-red-50 border-red-200 text-red-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800'
};

const alertIcons = {
    error: <XCircleIcon className="h-5 w-5 text-red-400" aria-hidden="true" />,
    warning: <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400" aria-hidden="true" />,
    info: <InformationCircleIcon className="h-5 w-5 text-blue-400" aria-hidden="true" />
};

const ErrorAlert: React.FC<ErrorAlertProps> = ({
                                                   message,
                                                   type = 'error',
                                                   onClose,
                                                   showIcon = true,
                                                   className = ''
                                               }) => {
    const styles = alertStyles[type] || alertStyles.error;
    const icon = alertIcons[type] || alertIcons.error;

    return (
        <div
            role="alert"
            aria-live="assertive"
            className={`border rounded-md p-4 flex items-start gap-3 ${styles} ${className}`}
        >
            {showIcon && (
                <div className="flex-shrink-0 mt-0.5">
                    {icon}
                </div>
            )}
            <div className="flex-1 text-sm font-medium">
                {message}
            </div>
            {onClose && (
                <button
                    type="button"
                    onClick={onClose}
                    aria-label="Đóng cảnh báo"
                    className="ml-auto p-1 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition"
                >
                    <XMarkIcon className="h-5 w-5 text-current" aria-hidden="true" />
                </button>
            )}
        </div>
    );
};

export { ErrorAlert };
export default ErrorAlert;
