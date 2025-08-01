import React, { HTMLAttributes, forwardRef, ReactNode } from 'react';
import { cn } from '@/utils/helpers';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'outlined' | 'elevated';
  children: ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hover?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      className,
      variant = 'default',
      children,
      padding = 'md',
      hover = false,
      ...props
    },
    ref
  ) => {
    const baseStyles = 'bg-white rounded-lg';

    const variants = {
      default: 'shadow-md',
      outlined: 'border border-gray-200',
      elevated: 'shadow-lg',
    };

    const paddings = {
      none: '',
      sm: 'p-3',
      md: 'p-6',
      lg: 'p-8',
    };

    return (
      <div
        className={cn(
          baseStyles,
          variants[variant],
          paddings[padding],
          hover && 'hover:shadow-md transition-shadow',
          className
        )}
        ref={ref}
        {...props}
      >
        {children}
      </div>
    );
  }
);

Card.displayName = 'Card';

export { Card };
export default Card;
