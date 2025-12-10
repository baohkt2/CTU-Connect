'use client';

import React, { useEffect, useState } from 'react';
import { User } from '@/types';
import { userService } from '@/services/userService';
import { toast } from 'react-hot-toast';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { UserPlus, Users, Search, Filter, X } from 'lucide-react';

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
      
      const response = await userService.searchFriendSuggestions(params);
      setSuggestions(response || []);
    } catch (err: any) {
      setError('Failed to load friend suggestions');
      console.error('Error loading friend suggestions:', err);
      toast.error('Failed to load suggestions');
    } finally {
      setLoading(false);
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
      toast.success('Friend request sent');
    } catch (err) {
      toast.error('Failed to send friend request');
      console.error('Error sending friend request:', err);
    } finally {
      setSendingRequest(null);
    }
  };

  const getConnectionBadges = (suggestion: any) => {
    const badges: string[] = [];
    if (suggestion.sameCollege) badges.push('Same College');
    if (suggestion.sameFaculty) badges.push('Same Faculty');
    if (suggestion.sameBatch) badges.push('Same Batch');
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
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with Title */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">
          {hasActiveFilters ? 'Search Results' : 'People You May Know'}
        </h3>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
        >
          <Filter className="w-4 h-4" />
          {showFilters ? 'Hide Filters' : 'Show Filters'}
        </button>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-lg border p-4 shadow-sm space-y-3">
        {/* Search Box */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Tìm theo tên hoặc mã số sinh viên..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <button
            onClick={handleSearch}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Search
          </button>
          {hasActiveFilters && (
            <button
              onClick={handleClearSearch}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors flex items-center gap-2"
            >
              <X className="w-4 h-4" />
              Clear
            </button>
          )}
        </div>

        {/* Filters (collapsible) */}
        {showFilters && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-3 border-t">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Faculty
              </label>
              <select
                value={filters.faculty}
                onChange={(e) => setFilters({...filters, faculty: e.target.value})}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loadingCategories}
              >
                <option value="">All Faculties</option>
                {faculties.map((faculty: any) => (
                  <option key={faculty.code} value={faculty.name}>
                    {faculty.name}
                  </option>
                ))}
              </select>
              {loadingCategories && (
                <p className="text-xs text-gray-500 mt-1">Loading faculties...</p>
              )}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Batch (Year)
              </label>
              <select
                value={filters.batch}
                onChange={(e) => setFilters({...filters, batch: e.target.value})}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loadingCategories}
              >
                <option value="">All Batches</option>
                {batches.map((batch: any) => (
                  <option key={batch.year} value={batch.year}>
                    {batch.year}
                  </option>
                ))}
              </select>
              {loadingCategories && (
                <p className="text-xs text-gray-500 mt-1">Loading batches...</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                College
              </label>
              <select
                value={filters.college}
                onChange={(e) => setFilters({...filters, college: e.target.value})}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loadingCategories}
              >
                <option value="">All Colleges</option>
                {colleges.map((college: any) => (
                  <option key={college.code} value={college.name}>
                    {college.name}
                  </option>
                ))}
              </select>
              {loadingCategories && (
                <p className="text-xs text-gray-500 mt-1">Loading colleges...</p>
              )}
            </div>
          </div>
        )}

        {/* Active Filters Display */}
        {hasActiveFilters && (
          <div className="flex flex-wrap gap-2 pt-2">
            {searchQuery && (
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                Query: {searchQuery}
              </span>
            )}
            {filters.faculty && (
              <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
                Faculty: {filters.faculty}
              </span>
            )}
            {filters.batch && (
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                Batch: {filters.batch}
              </span>
            )}
            {filters.college && (
              <span className="px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-sm">
                College: {filters.college}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Results */}
      {suggestions.length === 0 ? (
        <div className="text-center py-8 bg-white rounded-lg border">
          <Users className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500">
            {hasActiveFilters 
              ? 'No users found matching your search criteria' 
              : 'No friend suggestions available'}
          </p>
          {hasActiveFilters && (
            <button
              onClick={handleClearSearch}
              className="mt-3 text-blue-500 hover:text-blue-600"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <>
          <p className="text-sm text-gray-600">
            {suggestions.length} {suggestions.length === 1 ? 'person' : 'people'} found
            {!hasActiveFilters && ' (sorted by priority: same college → faculty → batch)'}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {suggestions.map((suggestion: any) => {
              const badges = getConnectionBadges(suggestion);
              
              return (
                <div key={suggestion.id} className="bg-white rounded-lg border p-4 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex flex-col items-center space-y-3">
                    <div className="w-16 h-16 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden">
                      {suggestion.avatarUrl ? (
                        <img
                          src={suggestion.avatarUrl}
                          alt={suggestion.fullName}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-gray-500 text-xl">
                          {suggestion.fullName?.charAt(0).toUpperCase() || 'U'}
                        </span>
                      )}
                    </div>

                    <div className="text-center w-full">
                      <h4 className="font-semibold text-gray-900">{suggestion.fullName}</h4>
                      <p className="text-sm text-gray-500">@{suggestion.username || suggestion.email?.split('@')[0]}</p>
                      
                      {/* Academic Info */}
                      {(suggestion.faculty || suggestion.major) && (
                        <div className="mt-2 space-y-1">
                          {suggestion.faculty && (
                            <p className="text-xs text-gray-600">{suggestion.faculty}</p>
                          )}
                          {suggestion.major && (
                            <p className="text-xs text-gray-500">{suggestion.major}</p>
                          )}
                          {suggestion.batch && (
                            <p className="text-xs text-gray-500">K{suggestion.batch}</p>
                          )}
                        </div>
                      )}

                      {/* Connection Badges */}
                      {badges.length > 0 && (
                        <div className="flex flex-wrap gap-1 justify-center mt-2">
                          {badges.map((badge, idx) => (
                            <span 
                              key={idx}
                              className="px-2 py-0.5 bg-blue-50 text-blue-600 text-xs rounded-full"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Mutual Friends */}
                      {suggestion.mutualFriendsCount && suggestion.mutualFriendsCount > 0 && (
                        <p className="text-xs text-blue-600 mt-2">
                          <Users className="w-3 h-3 inline mr-1" />
                          {suggestion.mutualFriendsCount} mutual {suggestion.mutualFriendsCount === 1 ? 'friend' : 'friends'}
                        </p>
                      )}
                    </div>

                    <button
                      onClick={() => handleSendRequest(suggestion.id)}
                      disabled={sendingRequest === suggestion.id}
                      className="w-full px-3 py-2 bg-blue-500 text-white rounded-lg text-sm hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-colors"
                    >
                      <UserPlus className="w-4 h-4" />
                      <span>{sendingRequest === suggestion.id ? 'Sending...' : 'Add Friend'}</span>
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
