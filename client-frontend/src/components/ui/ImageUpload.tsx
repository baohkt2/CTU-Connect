import {ChangeEvent, useRef, useState} from 'react';
import Image from 'next/image';
import {Image as ImageIcon, Upload, XCircle} from 'lucide-react';
import {useUploadMedia} from '@/hooks/useUploadMedia';

interface ImageUploadProps {
    currentUser: { id: string };
    currentImageUrl?: string;
    onImageUploaded: (url: string) => void;
    aspectRatio?: string;
    imageType: 'avatar' | 'background';
    maxSizeMB?: number;
}

export default function ImageUpload({
                                        currentUser,
                                        currentImageUrl,
                                        onImageUploaded,
                                        aspectRatio = '1/1',
                                        imageType = 'avatar',
                                        maxSizeMB = 5
                                    }: ImageUploadProps) {
    const [previewUrl, setPreviewUrl] = useState<string | null>(currentImageUrl || null);
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const {uploadMedia, status, progress, isUploading} = useUploadMedia();

    const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Check file size
        if (file.size > maxSizeMB * 1024 * 1024) {
            alert(`Kích thước tệp quá lớn. Vui lòng chọn tệp nhỏ hơn ${maxSizeMB}MB.`);
            return;
        }

        // Create temporary preview
        const objectUrl = URL.createObjectURL(file);
        setPreviewUrl(objectUrl);

        // Upload to server
        const response = await uploadMedia(file, currentUser.id, `Ảnh ${imageType}`);

        // If upload successful, pass URL to parent component
        if (response) {
            onImageUploaded(response.cloudinaryUrl);
        }

        // Clean up preview URL
        return () => URL.revokeObjectURL(objectUrl);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const file = e.dataTransfer.files?.[0];
        if (!file) return;

        // Check file size
        if (file.size > maxSizeMB * 1024 * 1024) {
            alert(`Kích thước tệp quá lớn. Vui lòng chọn tệp nhỏ hơn ${maxSizeMB}MB.`);
            return;
        }

        // Create temporary preview
        const objectUrl = URL.createObjectURL(file);
        setPreviewUrl(objectUrl);

        // Upload to server
        const response = await uploadMedia(file, imageType);

        // If upload successful, pass URL to parent component
        if (response) {
            onImageUploaded(response.cloudinaryUrl);
        }

        // Clean up preview URL
        return () => URL.revokeObjectURL(objectUrl);
    };

    const handleButtonClick = () => {
        fileInputRef.current?.click();
    };

    const handleRemoveImage = () => {
        setPreviewUrl(null);
        onImageUploaded('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="w-full">
            <div
                className={`relative border-2 border-dashed rounded-lg p-4 flex flex-col items-center justify-center transition-colors ${
                    isDragging
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-300 hover:border-blue-400'
                }`}
                style={{
                    aspectRatio: aspectRatio,
                    maxHeight: imageType === 'avatar' ? '200px' : '300px',
                }}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                {isUploading && (
                    <div
                        className="absolute inset-0 bg-white bg-opacity-80 flex flex-col items-center justify-center z-10">
                        <div className="w-full max-w-xs bg-gray-200 rounded-full h-2.5 mb-2">
                            <div
                                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                                style={{width: `${progress}%`}}
                            ></div>
                        </div>
                        <p className="text-sm text-gray-600">{`Đang tải lên... ${progress}%`}</p>
                    </div>
                )}

                {previewUrl ? (
                    <div className="relative w-full h-full">
                        <img
                            src={previewUrl}
                            alt="Preview"
                            className={`object-cover w-full h-full ${imageType === 'avatar' ? 'rounded-full' : 'rounded-lg'}`}
                        />
                        <button
                            type="button"
                            onClick={handleRemoveImage}
                            className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                            title="Xóa ảnh"
                        >
                            <XCircle size={20} />
                        </button>
                    </div>
                ) : (
                    <div className="text-center">
                        <ImageIcon className="mx-auto h-12 w-12 text-gray-400"/>
                        <div className="mt-2">
                            <button
                                type="button"
                                onClick={handleButtonClick}
                                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <Upload className="inline-block mr-1 h-4 w-4"/>
                                Chọn ảnh
                            </button>
                        </div>
                        <p className="mt-2 text-xs text-gray-500">
                            {imageType === 'avatar' ? 'Ảnh đại diện' : 'Ảnh bìa'}
                        </p>
                        <p className="text-xs text-gray-500">
                            PNG, JPG, GIF (tối đa {maxSizeMB}MB)
                        </p>
                    </div>
                )}
            </div>

            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
            />
        </div>
    );
}
