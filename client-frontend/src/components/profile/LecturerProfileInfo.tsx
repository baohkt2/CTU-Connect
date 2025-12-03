'use client';

import React from 'react';
import { LecturerProfile } from '@/types';
import {
  Briefcase,
  Award,
  MapPin,
  Hash,
  BookOpen,
  Star,
  Building
} from 'lucide-react';

interface LecturerProfileInfoProps {
  user: LecturerProfile;
}

export const LecturerProfileInfo: React.FC<LecturerProfileInfoProps> = ({ user }) => {
  // Define the item type
  type InfoItem = {
    label: string;
    value: string;
    icon: React.ReactElement;
  };

  const infoSections = [
    {
      title: 'Thông tin công việc',
      icon: <Briefcase className="h-5 w-5 text-blue-500" />,
      items: [
        user.staffCode && { label: 'Mã cán bộ', value: user.staffCode, icon: <Hash className="h-4 w-4" /> },
        user.position && { label: 'Chức vụ', value: user.position.name, icon: <Star className="h-4 w-4" /> },
        user.faculty && { label: 'Khoa', value: user.faculty.name, icon: <Building className="h-4 w-4" /> },
        user.college && { label: 'Trường', value: user.college.name, icon: <MapPin className="h-4 w-4" /> },
      ].filter((item): item is InfoItem => Boolean(item))
    },
    {
      title: 'Trình độ học vấn',
      icon: <Award className="h-5 w-5 text-green-500" />,
      items: [
        user.degree && { label: 'Bằng cấp', value: user.degree.name, icon: <Award className="h-4 w-4" /> },
        user.academic && { label: 'Học hàm/Học vị', value: user.academic.name, icon: <BookOpen className="h-4 w-4" /> },
      ].filter((item): item is InfoItem => Boolean(item))
    }
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-xl font-bold text-gray-900 mb-6 vietnamese-text">Thông tin cá nhân</h2>

      <div className="space-y-6">
        {infoSections.map((section, sectionIndex) => (
          <div key={sectionIndex}>
            <div className="flex items-center space-x-2 mb-4">
              {section.icon}
              <h3 className="font-semibold text-gray-800 vietnamese-text">{section.title}</h3>
            </div>

            <div className="space-y-3">
              {section.items.map((item, itemIndex) => (
                <div key={itemIndex} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                  <div className="text-gray-500">
                    {item.icon}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm text-gray-600 vietnamese-text">{item.label}</div>
                    <div className="font-medium text-gray-900 vietnamese-text">{item.value}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
