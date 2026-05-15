import pandas as pd
import os
import re

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dem")
ORTHO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "hirise_browse")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "mars_terrain_vault.csv")

# Mapping IDs to your 7 Thesis Categories
CAT_MAP = {
    # Craters
    "033968": "Craters", "089104": "Craters", "076968": "Craters", 
    "010245": "Craters", "003156": "Craters", "017719": "Craters", "003125": "Craters",
    # Gullies
    "025322": "Gullies", "022986": "Gullies", "010109": "Gullies", "017218": "Gullies",
    # Dunes
    "039932": "Dunes", "022534": "Dunes", 
    # Polar
    "005721": "Polar", "048808": "Polar", "049106": "Polar", "023009": "Polar", "010109": "Polar",
    # Volcanic
    "060706": "Volcanic", "006712": "Volcanic", "064700": "Volcanic",
    # Canyon / Chaos / Valleys
    "082989": "Canyon Walls", "059382": "Canyon Walls", "074396": "Canyon Walls",
    "074900": "Canyon Walls", "088616": "Canyon Walls", "029246": "Valleys",
    "042725": "Valleys", "033682": "Valleys"
}

def full_sync():
    print(f"🔍 Scanning DEM Folder: {DEM_DIR}")
    print(f"🔍 Scanning Ortho Folder: {ORTHO_DIR}")
    if not os.path.exists(DEM_DIR) or not os.path.exists(ORTHO_DIR):
        print("❌ ERROR: Data folders not found. Check the paths!")
        return

    # 1. Cleanup duplicates
    for directory in [DEM_DIR, ORTHO_DIR]:
        for f in os.listdir(directory):
            if "(1)" in f:
                os.remove(os.path.join(directory, f))
                print(f"🗑️ Deleted duplicate: {f}")

    # 2. Re-scan clean folders
    dtm_files = [f for f in os.listdir(DEM_DIR) if f.upper().endswith('.IMG')]
    ortho_files = [f for f in os.listdir(ORTHO_DIR) if f.upper().endswith('.JP2') or f.upper().endswith('.TIF')]
    
    vault_data = []
    
    print(f"📦 Found {len(dtm_files)} DTMs and {len(ortho_files)} Orthos. Matching now...")

    for dtm in dtm_files:
        # Extract orbit IDs (the 6-digit numbers)
        orbits = re.findall(r'\d{6}', dtm)
        if not orbits: continue
        
        main_orbit = orbits[0]
        
        # Find matching ortho (match ANY orbit ID found in the DTM filename)
        match_ortho = None
        for ortho in ortho_files:
            if any(orb in ortho for orb in orbits):
                match_ortho = ortho
                break
        
        if match_ortho:
            category = CAT_MAP.get(main_orbit, "Uncategorized")
            vault_data.append({
                "alias": f"{category}_{main_orbit}",
                "dem_id": dtm[:20],
                "terrain": category,
                "scale": 2.0 if "DTEED" in dtm else 1.0,
                "dtm_path": dtm,
                "ortho_path": match_ortho,
                "status": "Pending"
            })
            print(f"✅ Found Pair: {main_orbit} ({category})")
        else:
            print(f"⚠️ No Ortho partner found for {dtm}")

    # 3. Save to CSV
    try:
        df = pd.DataFrame(vault_data)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n✨ SUCCESS: Vault saved with {len(df)} sites.")
        print(f"📍 Location: {CSV_PATH}")
    except PermissionError:
        print("❌ ERROR: CSV is open in Excel! Close it and try again.")

if __name__ == "__main__":
    full_sync()
