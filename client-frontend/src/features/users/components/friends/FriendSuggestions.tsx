/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import React, { useEffect, useState } from 'react';
import { User } from '@/types';
import { userService } from '@/services/userService';
import { toast } from 'react-hot-toast';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { UserPlus, Users, Search, Filter, X, RefreshCw } from 'lucide-react';

interface Filters {
  college: string;
  faculty: string;
  batch: string;
}

interface FacultyOption {
  code: string;
  name: string;
}

interface BatchOption {
  year: string;
}

interface CollegeOption {
  code: string;
  name: string;
}

export const FriendSuggestions: React.FC = () => {
  const [suggestions, setSuggestions] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sendingRequest, setSendingRequest] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<Filters>({
    college: '',
    faculty: '',
    batch: ''
  });

  // Category options from database
  const [colleges, setColleges] = useState<CollegeOption[]>([]);
  const [faculties, setFaculties] = useState<FacultyOption[]>([]);
  const [batches, setBatches] = useState<BatchOption[]>([]);
  const [loadingCategories, setLoadingCategories] = useState(true);

  useEffect(() => {
    loadCategories();
    loadSuggestions();
  }, []); // Only load on mount

  const loadCategories = async () => {
    try {
      setLoadingCategories(true);
      const [collegesData, facultiesData, batchesData] = await Promise.all([
        userService.getColleges(),
        userService.getFaculties(),
        userService.getBatches()
      ]);
      
      setColleges(collegesData || []);
      setFaculties(facultiesData || []);
      setBatches(batchesData || []);
    } catch (err) {
      console.error('Error loading categories:', err);
      // Don't show error to user, just use empty arrays
    } finally {
      setLoadingCategories(false);
    }
  };

  const loadSuggestions = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params: any = { limit: 50 };
      
      // Add search query if exists
      if (searchQuery.trim()) {
        params.query = searchQuery.trim();
      }
      
      // Add filters if set
      if (filters.college) params.college = filters.college;
      if (filters.faculty) params.faculty = filters.faculty;
      if (filters.batch) params.batch = filters.batch;
      
      // Check if any filters are present
      const hasFilters = searchQuery.trim() || filters.college || filters.faculty || filters.batch;
      
      let response;
      if (hasFilters) {
        // Use search endpoint with filters
        response = await userService.searchFriendSuggestions(params);
      } else {
        // Use ML endpoint
        response = await userService.getFriendSuggestions(50);
      }
      
      setSuggestions(response || []);
    } catch (err: any) {
      setError('Không thể tải gợi ý kết bạn');
      console.error('Error loading friend suggestions:', err);
      toast.error('Không thể tải gợi ý kết bạn');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      // Invalidate cache and reload
      await userService.refreshFriendSuggestions();
      await loadSuggestions();
      toast.success('Đã làm mới danh sách gợi ý');
    } catch (err) {
      console.error('Error refreshing suggestions:', err);
      toast.error('Không thể làm mới gợi ý');
    } finally {
      setRefreshing(false);
    }
  };

  const handleSearch = () => {
    loadSuggestions();
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setFilters({ college: '', faculty: '', batch: '' });
    setTimeout(() => loadSuggestions(), 100);
  };

  const handleSendRequest = async (userId: string) => {
    try {
      setSendingRequest(userId);
      await userService.sendFriendRequest(userId);
      setSuggestions(prev => prev.filter(user => user.id !== userId));
      toast.success('Đã gửi lời mời kết bạn');
    } catch (err) {
      toast.error('Không thể gửi lời mời kết bạn');
      console.error('Error sending friend request:', err);
    } finally {
      setSendingRequest(null);
    }
  };

  const getConnectionBadges = (suggestion: any) => {
    const badges: string[] = [];
    if (suggestion.sameCollege) badges.push('Cùng trường');
    if (suggestion.sameFaculty) badges.push('Cùng khoa');
    if (suggestion.sameBatch) badges.push('Cùng khóa');
    return badges;
  };

  const hasActiveFilters = searchQuery.trim() || filters.college || filters.faculty || filters.batch;

  if (loading) {
    return (
      <div className="flex justify-center items-center py-8">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-500">{error}</p>
        <button
          onClick={loadSuggestions}
          className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Thử lại
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-3 sm:space-y-4">
      {/* Header with Title and Refresh Button */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 px-2 sm:px-0">
        <h3 className="text-base sm:text-lg font-semibold">
          {hasActiveFilters ? 'Kết quả tìm kiếm' : 'Những người bạn có thể biết'}
        </h3>
        <div className="flex items-center gap-2">
          {/* Refresh Button */}
          <button
            onClick={handleRefresh}
            disabled={refreshing || loading}
            className="flex items-center gap-2 px-3 py-1.5 text-xs sm:text-sm bg-green-100 hover:bg-green-200 text-green-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3 h-3 sm:w-4 sm:h-4 ${refreshing ? 'animate-spin' : ''}`} />
            {refreshing ? 'Đang làm mới...' : 'Làm mới'}
          </button>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2 px-3 py-1.5 text-xs sm:text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            <Filter className="w-3 h-3 sm:w-4 sm:h-4" />
            {showFilters ? 'Ẩn bộ lọc' : 'Hiện bộ lọc'}
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-lg border p-3 sm:p-4 shadow-sm space-y-3">
        {/* Search Box */}
        <div className="flex flex-col sm:flex-row gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4 sm:w-5 sm:h-5" />
            <input
              type="text"
              placeholder="Tìm theo tên hoặc mã số sinh viên..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              className="w-full pl-9 sm:pl-10 pr-3 sm:pr-4 py-2 text-sm sm:text-base border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <button
            onClick={handleSearch}
            className="px-4 sm:px-6 py-2 text-sm sm:text-base bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors whitespace-nowrap"
          >
            Tìm kiếm
          </button>
          {hasActiveFilters && (
            <button
              onClick={handleClearSearch}
              className="px-3 sm:px-4 py-2 text-sm sm:text-base bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors flex items-center justify-center gap-2"
            >
              <X className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Xóa</span>
            </button>
          )}
        </div>

        {/* Filters (collapsible) */}
        {showFilters && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-3 border-t">
            <div>
              <label className="block text-xs sm:text-sm font-medium text-gray-700 mb-1">
                Khoa
              </label>
              <select
                value={filters.faculty}
                onChange={(e) => setFilters({...filters, faculty: e.target.value})}
                className="w-full px-2 sm:px-3 py-1.5 sm:py-2 text-sm sm:text-base border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loadingCategories}
              >
                <option value="">Tất cả khoa</option>
                {faculties.map((faculty: any) => (
                  <option key={faculty.code} value={faculty.name}>
                    {faculty.name}
                  </option>
                ))}
              </select>
              {loadingCategories && (
                <p className="text-xs text-gray-500 mt-1">Đang tải khoa...</p>
              )}
            </div>
            
            <div>
              <label className="block text-xs sm:text-sm font-medium text-gray-700 mb-1">
                Khóa học
              </label>
              <select
                value={filters.batch}
                onChange={(e) => setFilters({...filters, batch: e.target.value})}
                className="w-full px-2 sm:px-3 py-1.5 sm:py-2 text-sm sm:text-base border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loadingCategories}
              >
                <option value="">Tất cả khóa</option>
                {batches.map((batch: any) => (
                  <option key={batch.year} value={batch.year}>
                    {batch.year}
                  </option>
                ))}
              </select>
              {loadingCategories && (
                <p className="text-xs text-gray-500 mt-1">Đang tải khóa học...</p>
              )}
            </div>

            <div>
              <label className="block text-xs sm:text-sm font-medium text-gray-700 mb-1">
                Trường
              </label>
              <select
                value={filters.college}
                onChange={(e) => setFilters({...filters, college: e.target.value})}
                className="w-full px-2 sm:px-3 py-1.5 sm:py-2 text-sm sm:text-base border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loadingCategories}
              >
                <option value="">Tất cả trường</option>
                {colleges.map((college: any) => (
                  <option key={college.code} value={college.name}>
                    {college.name}
                  </option>
                ))}
              </select>
              {loadingCategories && (
                <p className="text-xs text-gray-500 mt-1">Đang tải trường...</p>
              )}
            </div>
          </div>
        )}

        {/* Active Filters Display */}
        {hasActiveFilters && (
          <div className="flex flex-wrap gap-1.5 sm:gap-2 pt-2">
            {searchQuery && (
              <span className="px-2 sm:px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs sm:text-sm">
                Tìm kiếm: {searchQuery}
              </span>
            )}
            {filters.faculty && (
              <span className="px-2 sm:px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs sm:text-sm">
                Khoa: {filters.faculty}
              </span>
            )}
            {filters.batch && (
              <span className="px-2 sm:px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs sm:text-sm">
                Khóa: {filters.batch}
              </span>
            )}
            {filters.college && (
              <span className="px-2 sm:px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-xs sm:text-sm">
                Trường: {filters.college}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Results */}
      {suggestions.length === 0 ? (
        <div className="text-center py-6 sm:py-8 bg-white rounded-lg border">
          <Users className="w-10 h-10 sm:w-12 sm:h-12 text-gray-300 mx-auto mb-3" />
          <p className="text-sm sm:text-base text-gray-500">
            {hasActiveFilters 
              ? 'Không tìm thấy người dùng phù hợp với tiêu chí tìm kiếm' 
              : 'Không có gợi ý kết bạn'}
          </p>
          {hasActiveFilters && (
            <button
              onClick={handleClearSearch}
              className="mt-3 text-sm sm:text-base text-blue-500 hover:text-blue-600"
            >
              Xóa bộ lọc
            </button>
          )}
          {!hasActiveFilters && (
            <button
              onClick={handleRefresh}
              className="mt-3 text-sm sm:text-base text-blue-500 hover:text-blue-600"
            >
              Làm mới gợi ý
            </button>
          )}
        </div>
      ) : (
        <>
          <p className="text-xs sm:text-sm text-gray-600 px-2 sm:px-0">
            Tìm thấy {suggestions.length} người
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 sm:gap-4">
            {suggestions.map((suggestion: any) => {
              const badges = getConnectionBadges(suggestion);
              
              return (
                <div 
                  key={suggestion.id || suggestion.userId} 
                  className="bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow overflow-hidden"
                >
                  {/* Connection badges at top */}
                  {badges.length > 0 && (
                    <div className="px-3 pt-2 flex flex-wrap gap-1">
                      {badges.map((badge, idx) => (
                        <span key={idx} className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full text-xs">
                          {badge}
                        </span>
                      ))}
                    </div>
                  )}
                  
                  {/* Card content - clickable for profile */}
                  <div 
                    onClick={() => window.location.href = `/profile/${suggestion.id || suggestion.userId}`}
                    className="p-3 sm:p-4 cursor-pointer"
                  >
                    <div className="flex flex-col items-center space-y-2 sm:space-y-3">
                      <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden hover:ring-2 hover:ring-blue-500 transition-all">
                        {suggestion.avatarUrl ? (
                          <img
                            src={suggestion.avatarUrl}
                            alt={suggestion.fullName}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <span className="text-gray-500 text-lg sm:text-xl">
                            {suggestion.fullName?.charAt(0).toUpperCase() || 'U'}
                          </span>
                        )}
                      </div>

                      <div className="text-center w-full">
                        <h4 className="font-semibold text-sm sm:text-base text-gray-900 truncate hover:text-blue-600 transition-colors">
                          {suggestion.fullName}
                        </h4>
                        <p className="text-xs sm:text-sm text-gray-500 truncate">
                          @{suggestion.username || suggestion.email?.split('@')[0]}
                        </p>
                        
                        {/* Mutual Friends */}
                        {suggestion.mutualFriendsCount > 0 && (
                          <p className="text-xs text-blue-600 mt-1">
                            <Users className="w-3 h-3 inline mr-1" />
                            {suggestion.mutualFriendsCount} bạn chung
                          </p>
                        )}
                        
                        {/* Academic Info - using correct field names from FriendSuggestionDTO */}
                        <div className="mt-1.5 space-y-0.5">
                          {(suggestion.facultyName || suggestion.faculty) && (
                            <p className="text-xs text-gray-600 truncate">
                              {suggestion.facultyName || suggestion.faculty}
                            </p>
                          )}
                          {(suggestion.majorName || suggestion.major) && (
                            <p className="text-xs text-gray-500 truncate">
                              {suggestion.majorName || suggestion.major}
                            </p>
                          )}
                          {(suggestion.batchYear || suggestion.batch || suggestion.studentId) && (
                            <p className="text-xs text-gray-500 truncate">
                              {[
                                suggestion.batchYear || suggestion.batch,
                                suggestion.studentId
                              ].filter(Boolean).join(' • ')}
                            </p>
                          )}
                        </div>
                        
                        {/* Suggestion reason */}
                        {suggestion.suggestionReason && (
                          <p className="text-xs text-gray-400 mt-1 truncate italic">
                            {suggestion.suggestionReason}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Action button */}
                  <div className="px-3 pb-3 sm:px-4 sm:pb-4 pt-0">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSendRequest(suggestion.id || suggestion.userId);
                      }}
                      disabled={sendingRequest === (suggestion.id || suggestion.userId)}
                      className="w-full px-3 py-2 bg-blue-500 text-white rounded-lg text-xs sm:text-sm hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-colors"
                    >
                      <UserPlus className="w-3 h-3 sm:w-4 sm:h-4" />
                      <span>{sendingRequest === (suggestion.id || suggestion.userId) ? 'Đang gửi...' : 'Thêm bạn'}</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};

export default FriendSuggestions;
