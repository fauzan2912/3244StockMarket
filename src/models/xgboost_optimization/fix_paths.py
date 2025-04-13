"""
Fix paths in all Python files to use relative paths instead of absolute paths.
This makes the code more portable across different systems.
"""

import os
import glob

def fix_paths():
    """
    Replace absolute paths with relative paths in all Python files.
    Also creates necessary directories.
    """
    print("Fixing paths in all Python files...")
    
    # Create necessary directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Get all Python files
    py_files = glob.glob("*.py")
    
    for file_path in py_files:
        # Skip this file to avoid self-modification
        if file_path == "fix_paths.py":
            continue
            
        # Read the file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Replace absolute paths with relative paths
        content = content.replace('/home/ubuntu/stock_prediction/results/', './results/')
        content = content.replace('/home/ubuntu/stock_prediction/data/', './data/')
        content = content.replace('/home/ubuntu/stock_prediction/', './')
        content = content.replace('/home/ubuntu/', './')
        
        # Write the updated content
        with open(file_path, 'w') as file:
            file.write(content)
        
        print(f"✅ Updated paths in {file_path}")
    
    print("All paths have been updated to use relative paths.")

if __name__ == "__main__":
    fix_paths()
