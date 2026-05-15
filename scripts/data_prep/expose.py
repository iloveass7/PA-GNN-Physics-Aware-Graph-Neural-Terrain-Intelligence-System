import os
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dem")
ORTHO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "hirise_browse")

def list_the_intruders():
    print(f"📂 Scanning DEM Dir: {DEM_DIR}")
    print(f"📂 Scanning Ortho Dir: {ORTHO_DIR}")
    
    all_files = []
    if os.path.exists(DEM_DIR):
        all_files.extend(os.listdir(DEM_DIR))
    if os.path.exists(ORTHO_DIR):
        all_files.extend(os.listdir(ORTHO_DIR))
        
    extensions = [os.path.splitext(f)[1].lower() for f in all_files]
    stats = Counter(extensions)

    print(f"\n📂 Total Files: {len(all_files)}")
    print("--- 📊 File Type Breakdown ---")
    for ext, count in stats.items():
        print(f" {ext if ext else 'No Ext'}: {count} files")

    print("\n--- 🚩 Suspicious/Extra Files ---")
    for f in all_files:
        # Flag duplicates and extra metadata
        if "(1)" in f or f.lower().endswith(('.lbl', '.xml', '.zip')):
            print(f" [EXTRA] {f}")
            
if __name__ == "__main__":
    list_the_intruders()
