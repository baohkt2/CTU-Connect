'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, Search, Hash } from 'lucide-react';
import { searchService } from '@/services/searchService';

interface TrendingSearchesProps {
  onTrendingClick?: (term: string) => void;
  className?: string;
}

const TrendingSearches: React.FC<TrendingSearchesProps> = ({
  onTrendingClick,
  className = ''
}) => {
  const [trendingTerms, setTrendingTerms] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTrendingTerms = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await searchService.getTrendingSearchTerms();
        setTrendingTerms(response.trendingTerms || []);
      } catch (err) {
        console.error('Error fetching trending terms:', err);
        setError('Không thể tải từ khóa phổ biến');
      } finally {
        setLoading(false);
      }
    };

    fetchTrendingTerms();
  }, []);

  if (loading) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-orange-500" />
          Tìm kiếm phổ biến
        </h3>
        <div className="space-y-2">
          {[...Array(5)].map((_, index) => (
            <div key={index} className="animate-pulse">
              <div className="h-6 bg-gray-300 rounded w-3/4"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !trendingTerms.length) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-orange-500" />
          Tìm kiếm phổ biến
        </h3>
        <p className="text-sm text-gray-500">Chưa có dữ liệu tìm kiếm phổ biến</p>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <TrendingUp className="w-5 h-5 mr-2 text-orange-500" />
        Tìm kiếm phổ biến
      </h3>

      <div className="space-y-2">
        {trendingTerms.map((term, index) => (
          <button
            key={term}
            onClick={() => onTrendingClick?.(term)}
            className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-gray-50 transition-colors group"
          >
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-6 h-6 text-xs font-semibold">
                {index < 3 ? (
                  <span className={`
                    ${index === 0 ? 'text-yellow-600' : ''}
                    ${index === 1 ? 'text-gray-500' : ''}
                    ${index === 2 ? 'text-amber-600' : ''}
                  `}>
                    #{index + 1}
                  </span>
                ) : (
                  <Hash className="w-4 h-4 text-gray-400" />
                )}
              </div>
              <span className="text-gray-900 group-hover:text-blue-600 transition-colors">
                {term}
              </span>
            </div>
            <Search className="w-4 h-4 text-gray-400 group-hover:text-blue-600 transition-colors" />
          </button>
        ))}
      </div>
    </div>
  );
};

export default TrendingSearches;
