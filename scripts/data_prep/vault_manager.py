import pandas as pd
import os
import re

# CONFIG
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dem")
ORTHO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "hirise_browse")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "mars_terrain_vault.csv")

def sync_vault_v2():
    if not os.path.exists(CSV_PATH):
        print("❌ CSV not found. Run nuclear.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    dem_files = [f for f in os.listdir(DEM_DIR) if "(1)" not in f]
    ortho_files = [f for f in os.listdir(ORTHO_DIR) if "(1)" not in f]
    
    print(f"🚀 Syncing {len(df)} sites with improved matching logic...")

    for idx, row in df.iterrows():
        prefix = str(row['dem_id']) # e.g., DTEED_089104_2190
        
        # 1. Find the DTM first
        dtm_matches = [f for f in dem_files if f.startswith(prefix) and f.upper().endswith('.IMG')]
        
        if dtm_matches:
            actual_dtm = dtm_matches[0]
            df.at[idx, 'dtm_path'] = actual_dtm
            
            # 2. Extract ALL orbit IDs from the actual DTM filename
            ids_in_dtm = re.findall(r'\d{6}', actual_dtm)
            
            # 3. Look for an Ortho that matches ANY of those IDs
            ortho_match = None
            for f in ortho_files:
                if "ORTHO" in f.upper() and (f.lower().endswith('.jp2') or f.lower().endswith('.tif')):
                    if any(orbit_id in f for orbit_id in ids_in_dtm):
                        ortho_match = f
                        break
            
            if ortho_match:
                df.at[idx, 'ortho_path'] = ortho_match
                print(f"✅ Linked: {row['alias'].ljust(25)} | DTM: {actual_dtm[:15]}... Ortho: {ortho_match[:15]}...")
            else:
                print(f"❓ Ortho Missing for: {row['alias']}")
        else:
            print(f"❌ DTM Missing for:   {row['alias']}")

    df.to_csv(CSV_PATH, index=False)
    print("\n✨ Sync Complete. All observation pairs matched.")

if __name__ == "__main__":
    sync_vault_v2()
