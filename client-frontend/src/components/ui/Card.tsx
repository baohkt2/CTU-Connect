import React, { HTMLAttributes, forwardRef, ReactNode } from 'react';
import { cn } from '@/utils/helpers';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'outlined' | 'elevated';
    children: ReactNode;
    className?: string;
    padding?: 'none' | 'sm' | 'md' | 'lg';
    hover?: boolean;
    as?: React.ElementType;
}

const Card = forwardRef<HTMLElement, CardProps>(
    (
        {
            as: Component = 'div',
            className,
            variant = 'default',
            children,
            padding = 'md',
            hover = false,
            ...props
        },
        ref
    ) => {
        const baseStyles = 'bg-white rounded-lg ';

        const variants = {
            default: 'shadow-md ',
            outlined: 'border border-gray-200 ',
            elevated: 'shadow-lg ',
        };

        const paddings = {
            none: '',
            sm: 'p-3',
            md: 'p-6',
            lg: 'p-8',
        };

        const classes = cn(
            baseStyles,
            variants[variant],
            paddings[padding],
            hover && 'transition-shadow duration-300 ease-in-out hover:shadow-xl',
            className
        );

        return (
            <Component
                className={classes}
                ref={ref}
                {...props}
            >
                {children}
            </Component>
        );
    }
);

Card.displayName = 'Card';

export { Card };
export default Card;
