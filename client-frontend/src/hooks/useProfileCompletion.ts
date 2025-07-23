import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { userService } from '@/services/userService';
import { User } from '@/types';

export const useProfileCompletion = (user?: User) => {
  const [isChecking, setIsChecking] = useState(true);
  const [isCompleted, setIsCompleted] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const checkProfileCompletion = async () => {
      if (!user) {
        setIsChecking(false);
        return;
      }

      try {
        const completed = await userService.checkProfileCompletion();
        setIsCompleted(completed);

        // If profile is not completed, redirect to update profile page
        if (!completed) {
          router.push('/profile/update');
        }
      } catch (error) {
        console.error('Error checking profile completion:', error);
        // If there's an error, assume profile needs completion
        setIsCompleted(false);
        router.push('/profile/update');
      } finally {
        setIsChecking(false);
      }
    };

    checkProfileCompletion();
  }, [user, router]);

  return { isChecking, isCompleted };
};
