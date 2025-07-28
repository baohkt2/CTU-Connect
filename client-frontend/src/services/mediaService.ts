import api from '@/lib/api';
import { ApiResponse } from '@/types';

export interface UploadResponse {
  id: string;                    // BE: id
  fileName: string;                 // BE: fileName
  originalFileName: string;         // BE: originalFileName
  cloudinaryUrl: string;            // BE: cloudinaryUrl
  cloudinaryPublicId: string;                 // BE: cloudinaryPublicId
  contentType: string;                 // BE: contentType
  mediaType: 'AVATAR' | 'BACKGROUND' | string;  // BE: mediaType enum
  fileSize: number;                 // BE: fileSize
  uploadedBy: string;               // BE: uploadedBy
  description?: string;             // BE: description
  createdAt?: string;               // BE: createdAt
  updatedAt?: string;               // BE: updatedAt
}


export const mediaService = {
  /**
   * Upload a file to the media service
   * @param file The file to upload
   * @param uploadedBy The user ID who uploads the file
   * @param description Optional description for the file
   * @returns The uploaded file metadata including URL
   */
  async uploadFile(
      file: File,
      uploadedBy: string,
      description?: string
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('uploadedBy', uploadedBy);
    if (description) {
      formData.append('description', description);
    }

    const response = await api.post('/media/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    console.log('DEBUG: Upload response:', response.data);
    return response.data;
  },

  /**
   * Get media by ID
   * @param id Media ID
   */
  async getMediaById(id: number): Promise<UploadResponse> {
    const response = await api.get(`/media/${id}`);
    return response.data;
  },

  /**
   * Get media files uploaded by a specific user
   * @param uploadedBy User ID
   */
  async getMediaByUser(uploadedBy: string): Promise<UploadResponse[]> {
    const response = await api.get(`/media/user/${uploadedBy}`);
    return response.data;
  },

  /**
   * Get media files by type
   * @param mediaType Type of media (e.g., AVATAR, BACKGROUND)
   */
  async getMediaByType(mediaType: string): Promise<UploadResponse[]> {
    const response = await api.get(`/media/type/${mediaType}`);
    return response.data;
  },

  /**
   * Search media by keyword
   * @param keyword Keyword to search for
   */
  async searchMedia(keyword: string): Promise<UploadResponse[]> {
    const response = await api.get('/media/search', {
      params: { keyword },
    });
    return response.data;
  },

  /**
   * Delete media by ID
   * @param id Media ID
   */
  async deleteMedia(id: number): Promise<ApiResponse<null>> {
    const response = await api.delete(`/media/${id}`);
    return response.data;
  },
};
