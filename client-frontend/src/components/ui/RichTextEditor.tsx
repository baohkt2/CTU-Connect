'use client';

import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import ReactQuill using react-quill-new for React 19 compatibility
const ReactQuill = dynamic(() => import('react-quill-new'), {
  ssr: false,
  loading: () => <div className="h-32 bg-gray-50 animate-pulse rounded-lg"></div>
});

// Import Quill styles
import 'react-quill-new/dist/quill.snow.css';

interface RichTextEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  maxLength?: number;
  disabled?: boolean;
}

export interface RichTextEditorRef {
  focus: () => void;
  getLength: () => number;
  getText: () => string;
}

const RichTextEditor = forwardRef<RichTextEditorRef, RichTextEditorProps>(({
  value,
  onChange,
  placeholder = "Bạn đang nghĩ gì? Hãy chia sẻ với cộng đồng CTU...",
  className = "",
  maxLength = 2000,
  disabled = false
}, ref) => {
  const quillRef = useRef<any>(null);

  useImperativeHandle(ref, () => ({
    focus: () => {
      if (quillRef.current && quillRef.current.getEditor) {
        quillRef.current.getEditor().focus();
      }
    },
    getLength: () => {
      if (quillRef.current && quillRef.current.getEditor) {
        return quillRef.current.getEditor().getLength() - 1; // Subtract 1 for the trailing newline
      }
      return 0;
    },
    getText: () => {
      if (quillRef.current && quillRef.current.getEditor) {
        return quillRef.current.getEditor().getText();
      }
      return '';
    }
  }));

  // Custom toolbar configuration - simplified to avoid format issues
  const modules = {
    toolbar: [
      [{ 'header': [1, 2, 3, false] }],
      ['bold', 'italic', 'underline', 'strike'],
      [{ 'color': [] }, { 'background': [] }],
      [{ 'list': 'ordered'}, { 'list': 'bullet' }],
      ['blockquote', 'code-block'],
      ['link'],
      [{ 'align': [] }],
      ['clean']
    ],
    clipboard: {
      matchVisual: false,
    }
  };

  // Correct format names for Quill - simplified list
  const formats = [
    'header',
    'bold', 'italic', 'underline', 'strike',
    'color', 'background',
    'list',
    'blockquote', 'code-block',
    'link',
    'align'
  ];

  // Handle text change and enforce max length
  const handleChange = (content: string, delta: any, source: any, editor: any) => {
    const text = editor.getText();
    const length = text.length - 1; // Subtract 1 for the trailing newline

    if (length <= maxLength) {
      onChange(content);
    } else {
      // If exceeding max length, truncate the content
      const truncatedText = text.substring(0, maxLength);
      const truncatedDelta = editor.clipboard.convert(truncatedText);
      editor.setContents(truncatedDelta);
    }
  };

  // Get current text length for display
  const getCurrentLength = () => {
    if (quillRef.current && quillRef.current.getEditor) {
      return Math.max(0, quillRef.current.getEditor().getLength() - 1);
    }
    return 0;
  };

  return (
    <div className={`rich-text-editor ${className}`}>
      <ReactQuill
        ref={quillRef}
        theme="snow"
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        modules={modules}
        formats={formats}
        readOnly={disabled}
        style={{
          backgroundColor: disabled ? '#f9fafb' : 'white',
        }}
      />

      {/* Character counter */}
      <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
        <span>Hỗ trợ định dạng văn bản, danh sách, liên kết và nhiều hơn nữa</span>
        <span>
          {getCurrentLength()}/{maxLength}
        </span>
      </div>

      {/* Custom CSS for better styling */}
      <style jsx global>{`
        .rich-text-editor .ql-toolbar {
          border: 1px solid #e5e7eb;
          border-bottom: none;
          border-radius: 0.5rem 0.5rem 0 0;
          background: #f8fafc;
          padding: 8px;
        }
        
        .rich-text-editor .ql-container {
          border: 1px solid #e5e7eb;
          border-radius: 0 0 0.5rem 0.5rem;
          font-family: inherit;
          font-size: 14px;
          line-height: 1.6;
        }
        
        .rich-text-editor .ql-editor {
          min-height: 120px;
          padding: 16px;
          color: #374151;
        }
        
        .rich-text-editor .ql-editor.ql-blank::before {
          color: #9ca3af;
          font-style: normal;
        }
        
        .rich-text-editor:focus-within .ql-toolbar,
        .rich-text-editor:focus-within .ql-container {
          border-color: #6366f1;
        }
        
        .rich-text-editor .ql-toolbar .ql-picker-label {
          color: #4b5563;
        }
        
        .rich-text-editor .ql-toolbar .ql-stroke {
          stroke: #4b5563;
        }
        
        .rich-text-editor .ql-toolbar .ql-fill {
          fill: #4b5563;
        }
        
        .rich-text-editor .ql-toolbar button:hover {
          color: #6366f1;
        }
        
        .rich-text-editor .ql-toolbar button:hover .ql-stroke {
          stroke: #6366f1;
        }
        
        .rich-text-editor .ql-toolbar button:hover .ql-fill {
          fill: #6366f1;
        }
        
        .rich-text-editor .ql-toolbar .ql-active {
          color: #6366f1;
        }
        
        .rich-text-editor .ql-toolbar .ql-active .ql-stroke {
          stroke: #6366f1;
        }
        
        .rich-text-editor .ql-toolbar .ql-active .ql-fill {
          fill: #6366f1;
        }

        /* Custom styles for better mobile experience */
        @media (max-width: 768px) {
          .rich-text-editor .ql-toolbar {
            padding: 6px;
          }
          
          .rich-text-editor .ql-editor {
            padding: 12px;
            min-height: 100px;
          }
        }
      `}</style>
    </div>
  );
});

RichTextEditor.displayName = 'RichTextEditor';

export default RichTextEditor;
