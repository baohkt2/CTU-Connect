'use client';

import React from 'react';
import { User } from '@/types';
import { 
  GraduationCap, 
  BookOpen, 
  Calendar, 
  Users, 
  MapPin,
  Award,
  Hash
} from 'lucide-react';

interface StudentProfileInfoProps {
  user: User;
}

export const StudentProfileInfo: React.FC<StudentProfileInfoProps> = ({ user }) => {
  const infoSections = [
    {
      title: 'Thông tin học tập',
      icon: <GraduationCap className="h-5 w-5 text-blue-500" />,
      items: [
        user.studentId && { label: 'Mã số sinh viên', value: user.studentId, icon: <Hash className="h-4 w-4" /> },
        user.major && { label: 'Ngành học', value: user.major.name, icon: <BookOpen className="h-4 w-4" /> },
        user.faculty && { label: 'Khoa', value: user.faculty.name, icon: <Award className="h-4 w-4" /> },
        user.college && { label: 'Trường', value: user.college.name, icon: <MapPin className="h-4 w-4" /> },
        user.yearOfStudy && { label: 'Năm học', value: `Năm ${user.yearOfStudy}`, icon: <Calendar className="h-4 w-4" /> },
        user.batch && { label: 'Khóa', value: `Khóa ${user.batch.year}`, icon: <Users className="h-4 w-4" /> },
      ].filter(Boolean)
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
