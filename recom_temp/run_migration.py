#!/usr/bin/env python3
"""
Script đơn giản để chạy migration dữ liệu
Sử dụng: python run_migration.py
"""

import os
import sys
import subprocess

def check_requirements():
    """Kiểm tra và cài đặt requirements"""
    try:
        import pandas
        import psycopg2
        import pymongo
        import neo4j
        import bcrypt
        print("✅ Tất cả thư viện đã được cài đặt")
        return True
    except ImportError as e:
        print(f"❌ Thiếu thư viện: {e}")
        print("🔧 Đang cài đặt requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Đã cài đặt thành công requirements")
            return True
        except subprocess.CalledProcessError:
            print("❌ Không thể cài đặt requirements")
            return False

def check_databases():
    """Kiểm tra kết nối database"""
    print("🔍 Kiểm tra kết nối database...")

    # Check PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost', port=5434, database='auth_db',
            user='postgres', password='password'
        )
        conn.close()
        print("✅ PostgreSQL (auth_db) - OK")
    except Exception as e:
        print(f"❌ PostgreSQL (auth_db) - {e}")
        return False

    # Check MongoDB
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        client.admin.command('ismaster')
        client.close()
        print("✅ MongoDB (post_db) - OK")
    except Exception as e:
        print(f"❌ MongoDB (post_db) - {e}")
        return False

    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
        driver.verify_connectivity()
        driver.close()
        print("✅ Neo4j - OK")
    except Exception as e:
        print(f"❌ Neo4j - {e}")
        return False

    return True

def run_migration():
    """Chạy migration dữ liệu"""
    print("\n🚀 Bắt đầu migration dữ liệu...")

    try:
        from data_migration import DataMigration
        migration = DataMigration()
        migration.run_migration(n_users=200, n_posts=1000)
        print("\n🎉 Migration hoàn tất thành công!")

        # In thống kê
        print("\n📊 Thống kê dữ liệu đã migration:")
        print("   - PostgreSQL (auth_db): 200 users")
        print("   - Neo4j: 200 users + relationships + faculties + categories")
        print("   - MongoDB (post_db): 1000 posts")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình migration: {e}")
        return False

    return True

def main():
    print("🎯 CTU Connect - Data Migration Tool")
    print("=" * 50)

    # Kiểm tra requirements
    if not check_requirements():
        print("❌ Vui lòng cài đặt requirements trước khi tiếp tục")
        return

    # Kiểm tra databases
    if not check_databases():
        print("❌ Vui lòng khởi động tất cả databases trước khi migration")
        print("💡 Chạy: docker-compose up -d")
        return

    # Xác nhận migration
    response = input("\n❓ Bạn có muốn tiếp tục migration dữ liệu? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Đã hủy migration")
        return

    # Chạy migration
    if run_migration():
        print("\n✨ Dữ liệu đã sẵn sàng cho hệ thống CTU Connect!")
    else:
        print("\n❌ Migration thất bại. Vui lòng kiểm tra lại!")

if __name__ == "__main__":
    main()
