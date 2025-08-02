import { useState } from 'react';
import { mediaService, MediaResponse } from '@/services/mediaService';
import { useToast } from './useToast';

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

export function useUploadMedia() {
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [progress, setProgress] = useState(0);
  const { showToast } = useToast();

  const uploadMedia = async (
      file: File,
      description?: string
  ): Promise<MediaResponse | null> => {
    try {
      setStatus('uploading');
      setProgress(0);

      // Simple progress simulation
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + 5;
          return newProgress < 90 ? newProgress : prev;
        });
      }, 100);

      const response = await mediaService.uploadFile(file, description);

      clearInterval(progressInterval);
      setProgress(100);
      setStatus('success');

      return response;
    } catch (error) {
      setStatus('error');
      showToast('Không thể tải lên tệp. Vui lòng thử lại sau.', 'error');
      console.error('Upload error:', error);
      return null;
    }
  };

  const reset = () => {
    setStatus('idle');
    setProgress(0);
  };

  return {
    uploadMedia,
    status,
    progress,
    isUploading: status === 'uploading',
    isSuccess: status === 'success',
    isError: status === 'error',
    reset
  };
}
