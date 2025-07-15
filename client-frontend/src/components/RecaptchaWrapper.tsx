'use client';

import React from 'react';
import { GoogleReCaptchaProvider } from 'react-google-recaptcha-v3';
import { ClientOnly } from './ClientOnly';

interface RecaptchaWrapperProps {
  children: React.ReactNode;
}

export const RecaptchaWrapper: React.FC<RecaptchaWrapperProps> = ({ children }) => {
  const siteKey = process.env.NEXT_PUBLIC_RECAPTCHA_SITE_KEY;

  if (!siteKey) {
    console.warn('reCAPTCHA site key not found');
    return <>{children}</>;
  }

  return (
    <ClientOnly fallback={<>{children}</>}>
      <GoogleReCaptchaProvider
        reCaptchaKey={siteKey}
        scriptProps={{
          async: false,
          defer: false,
          appendTo: "head",
          nonce: undefined,
        }}
      >
        {children}
      </GoogleReCaptchaProvider>
    </ClientOnly>
  );
};
