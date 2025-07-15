import { useGoogleReCaptcha } from 'react-google-recaptcha-v3';
import { useCallback, useEffect, useState } from 'react';

export const useRecaptcha = () => {
  const { executeRecaptcha } = useGoogleReCaptcha();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    // Chỉ set isReady = true sau khi component đã mount trên client
    setIsReady(!!executeRecaptcha);
  }, [executeRecaptcha]);

  const executeRecaptchaAction = useCallback(
    async (action: string): Promise<string | null> => {
      if (!executeRecaptcha) {
        console.warn('reCAPTCHA not ready');
        return null;
      }

      try {
        const token = await executeRecaptcha(action);
        return token;
      } catch (error) {
        console.error('reCAPTCHA execution error:', error);
        return null;
      }
    },
    [executeRecaptcha]
  );

  return {
    executeRecaptcha: executeRecaptchaAction,
    isReady,
  };
};

// Define common action types
export const RECAPTCHA_ACTIONS = {
  LOGIN: 'login',
  REGISTER: 'register',
  FORGOT_PASSWORD: 'forgot_password',
  RESET_PASSWORD: 'reset_password',
  CONTACT: 'contact',
} as const;

export type RecaptchaAction = typeof RECAPTCHA_ACTIONS[keyof typeof RECAPTCHA_ACTIONS];
