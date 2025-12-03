/*
import { apiClient } from '@/shared/config/api-client';
import { storage } from '@/shared/utils';
import { ApiResponse, FileUploadResponse } from '@/shared/types';

/!**
 * Upload Service
 * Handles file uploads and media processing (cross-domain)
 *!/
export class UploadService {
  /!**
   * Upload single file
   *!/
  async uploadFile(
    file: File,
    endpoint: string,
    onProgress?: (progress: number) => void
  ): Promise<FileUploadResponse> {
    return apiClient.uploadFile<FileUploadResponse>(endpoint, file, onProgress);
  }

  /!**
   * Upload multiple files
   *!/
  async uploadFiles(
    files: File[],
    endpoint: string,
    onProgress?: (progress: number) => void
  ): Promise<FileUploadResponse[]> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    return apiClient.post<FileUploadResponse[]>(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(progress);
        }
      },
    });
  }

  /!**
   * Delete file
   *!/
  async deleteFile(fileId: string): Promise<ApiResponse<null>> {
    return apiClient.delete<ApiResponse<null>>(`/files/${fileId}`);
  }

  /!**
   * Get file metadata
   *!/
  async getFileMetadata(fileId: string): Promise<FileUploadResponse> {
    return apiClient.get<FileUploadResponse>(`/files/${fileId}/metadata`);
  }
}

// Export singleton instance
export const uploadService = new UploadService();
*/
