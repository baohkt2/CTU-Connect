import { useMutation, useQueryClient } from '@tanstack/react-query';
import { mediaService } from '@/services/mediaService';

export const useMediaHooks = () => {
  const queryClient = useQueryClient();

  // Upload single file
  const useUploadFile = () => {
    return useMutation({
      mutationFn: ({ file, uploadedBy, description }: { 
        file: File; 
        uploadedBy: string; 
        description?: string 
      }) => mediaService.uploadFile(file, uploadedBy, description),
      onError: (error) => {
        console.error('Error uploading file:', error);
      },
    });
  };

  // Upload multiple files
  const useUploadMultipleFiles = () => {
    return useMutation({
      mutationFn: ({ files, uploadedBy, description }: { 
        files: File[]; 
        uploadedBy: string; 
        description?: string 
      }) => mediaService.uploadMultipleFiles(files, uploadedBy, description),
      onError: (error) => {
        console.error('Error uploading multiple files:', error);
      },
    });
  };

  // Delete media
  const useDeleteMedia = () => {
    return useMutation({
      mutationFn: mediaService.deleteMedia,
      onError: (error) => {
        console.error('Error deleting media:', error);
      },
    });
  };

  return {
    useUploadFile,
    useUploadMultipleFiles,
    useDeleteMedia,
  };
}; 