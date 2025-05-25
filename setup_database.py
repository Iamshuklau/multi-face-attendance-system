#!/usr/bin/env python3
"""
Database Setup Script for Multi-Face Attendance System

This script helps you set up the PostgreSQL database and tables.
Run this before starting the application.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import DATABASE_CONFIG
from utils.db_utils import create_tables

def create_database():
    """Create the database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server (not the specific database)
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['username'],
            password=DATABASE_CONFIG['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DATABASE_CONFIG['database'],))
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(f"CREATE DATABASE {DATABASE_CONFIG['database']}")
            print(f"✅ Database '{DATABASE_CONFIG['database']}' created successfully!")
        else:
            print(f"ℹ️  Database '{DATABASE_CONFIG['database']}' already exists.")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Error creating database: {e}")
        return False
    
    return True

def setup_tables():
    """Create all tables using SQLAlchemy."""
    try:
        create_tables()
        print("✅ All tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def main():
    print("🚀 Setting up Multi-Face Attendance System Database...")
    print("-" * 50)
    
    # Display current configuration
    print(f"Host: {DATABASE_CONFIG['host']}")
    print(f"Port: {DATABASE_CONFIG['port']}")
    print(f"Database: {DATABASE_CONFIG['database']}")
    print(f"Username: {DATABASE_CONFIG['username']}")
    print("-" * 50)
    
    # Create database
    print("\n1. Creating database...")
    if not create_database():
        print("❌ Failed to create database. Please check your PostgreSQL configuration.")
        return
    
    # Create tables
    print("\n2. Creating tables...")
    if not setup_tables():
        print("❌ Failed to create tables. Please check your database connection.")
        return
    
    print("\n🎉 Database setup completed successfully!")
    print("\nNext steps:")
    print("1. Update your PostgreSQL credentials in config.py")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the app: streamlit run app.py")

if __name__ == "__main__":
    main()