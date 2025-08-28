#!/usr/bin/env python3
"""
Simple File Upload and Analysis Script
Upload your data files and get instant AI-powered insights!
"""

import os
import sys
from pathlib import Path

def main():
    print("🤖 AI Data Analyzer - File Upload Helper")
    print("=" * 50)
    
    # Get file path from user
    file_path = input("📁 Enter the path to your data file: ").strip().strip('"')
    
    if not file_path:
        print("❌ No file path provided")
        return
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Get analysis prompt
    print("\n💬 What would you like to analyze? Examples:")
    print("   • 'Describe this data'")
    print("   • 'Find correlations between all variables'")
    print("   • 'Predict [target column] using other features'")
    print("   • 'Find customer segments'")
    print("   • 'Detect anomalies'")
    
    prompt = input("\n🎯 Your analysis request: ").strip()
    
    if not prompt:
        prompt = "Describe this data"
    
    # Build command
    analyzer_path = "src/ai_data_analyzer.py"
    if not os.path.exists(analyzer_path):
        print("❌ AI Data Analyzer not found. Make sure you're in the correct directory.")
        return
    
    command = f'python {analyzer_path} --data "{file_path}" --prompt "{prompt}"'
    
    print(f"\n🚀 Running analysis...")
    print(f"📋 Command: {command}")
    print("-" * 50)
    
    # Execute analysis
    os.system(command)

if __name__ == "__main__":
    main()
