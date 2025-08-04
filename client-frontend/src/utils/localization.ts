import { DateTime } from 'luxon'
import { VI_LOCALE } from '@/lib/locales/vi'

export function t(key: string, params?: Record<string, string | number>): string {
  const keys = key.split('.')
  let value: any = VI_LOCALE

  for (const k of keys) {
    if (value && typeof value === 'object' && k in value) {
      value = value[k]
    } else {
      console.warn(`Translation key not found: ${key}`)
      return key
    }
  }

  if (typeof value !== 'string') {
    console.warn(`Translation value is not a string: ${key}`)
    return key
  }

  if (params) {
    return value.replace(/\{(\w+)\}/g, (_, paramKey) => {
      return params[paramKey]?.toString() ?? `{${paramKey}}`
    })
  }

  return value
}

/**
 * Format time relative to now in Vietnamese using Hanoi timezone
 * @param inputDate - ISO string or Date
 */
export function formatTimeAgo(inputDate: string | Date): string {
  const date = typeof inputDate === 'string'
      ? DateTime.fromISO(inputDate, { zone: 'utc' }).setZone('Asia/Ho_Chi_Minh')
      : DateTime.fromJSDate(inputDate).setZone('Asia/Ho_Chi_Minh')

  const now = DateTime.now().setZone('Asia/Ho_Chi_Minh')

  if (!date.isValid) return t('time.invalidDate')

  const diffInSeconds = Math.floor(now.diff(date, 'seconds').seconds)

  if (diffInSeconds < 60) {
    return t('time.now')
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60)
    return t('time.minutesAgo', { count: minutes })
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600)
    return t('time.hoursAgo', { count: hours })
  } else if (diffInSeconds < 604800) {
    const days = Math.floor(diffInSeconds / 86400)
    return t('time.daysAgo', { count: days })
  } else if (diffInSeconds < 2592000) {
    const weeks = Math.floor(diffInSeconds / 604800)
    return t('time.weeksAgo', { count: weeks })
  } else if (diffInSeconds < 31536000) {
    const months = Math.floor(diffInSeconds / 2592000)
    return t('time.monthsAgo', { count: months })
  } else {
    const years = Math.floor(diffInSeconds / 31536000)
    return t('time.yearsAgo', { count: years })
  }
}


/**
 * Format file size in Vietnamese
 * @param bytes - File size in bytes
 * @returns Formatted file size string
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Get category name in Vietnamese
 * @param categoryKey - Category key
 * @returns Vietnamese category name
 */
export function getCategoryName(categoryKey: string): string {
  return t(`categories.${categoryKey}`) || categoryKey;
}
