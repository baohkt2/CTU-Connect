import api from '@/lib/api';

export interface MediaResponse {
  id: string;
  fileName: string;
  originalFileName: string;
  cloudinaryUrl: string;
  cloudinaryPublicId: string;
  contentType: string;
  mediaType: 'IMAGE' | 'VIDEO' | 'DOCUMENT' | 'AUDIO';
  fileSize: number;
  description?: string;
  uploadedBy: string;
  createdAt: string;
  updatedAt: string;
}

export const mediaService = {
  // Upload single file to media service
  async uploadFile(file: File, description?: string): Promise<MediaResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (description) {
      formData.append('description', description);
    }

    const response = await api.post('/media/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Upload multiple files to media service
  async uploadFiles(files: File[] , description?: string): Promise<MediaResponse[]> {
    const uploadPromises = files.map(file =>
      this.uploadFile(file, description)
    );
    return Promise.all(uploadPromises);
  },

  // Get media by ID
  async getMediaById(id: string): Promise<MediaResponse> {
    const response = await api.get(`/media/${id}`);
    return response.data;
  },

  // Get media by user by uploadedBy field
  async getMediaByUser(uploadedBy: string): Promise<MediaResponse[]> {
    const response = await api.get(`/media/user/${uploadedBy}`);
    return response.data;
  },

    // Get media by me
    async getMediaByMe(): Promise<MediaResponse[]> {
    const response = await api.get('/media/me');
    return response.data;
    },



  // Delete media
  async deleteMedia(id: string): Promise<void> {
    await api.delete(`/media/${id}`);
  }
};
