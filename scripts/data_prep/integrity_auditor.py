import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dem")
ORTHO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "hirise_browse")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "mars_terrain_vault.csv")

def run_final_audit():
    if not os.path.exists(CSV_PATH):
        print("❌ CSV not found. Run nuclear.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    verified_count = 0

    print("🧐 Performing Final Integrity Audit...\n")

    for idx, row in df.iterrows():
        dtm_p = os.path.join(DEM_DIR, str(row['dtm_path']))
        ortho_p = os.path.join(ORTHO_DIR, str(row['ortho_path']))

        # Check if both paths are valid and actually exist
        if os.path.exists(dtm_p) and os.path.exists(ortho_p):
            # Check sizes - HiRISE files should be chunky (at least 5MB)
            dtm_size = os.path.getsize(dtm_p) / (1024*1024)
            ortho_size = os.path.getsize(ortho_p) / (1024*1024)

            if dtm_size > 5 and ortho_size > 5:
                df.at[idx, 'status'] = "Verified"
                verified_count += 1
                print(f"💎 Verified: {row['alias'].ljust(25)} | {dtm_size:.1f}MB / {ortho_size:.1f}MB")
            else:
                df.at[idx, 'status'] = "Corrupt/Small"
                print(f"⚠️ Warning:  {row['alias'].ljust(25)} | Files too small!")
        else:
            df.at[idx, 'status'] = "Missing"
            print(f"❌ Missing:   {row['alias'].ljust(25)}")

    df.to_csv(CSV_PATH, index=False)
    print(f"\n📊 TOTAL: {verified_count}/{len(df)} sites are ready for GNN Training.")

if __name__ == "__main__":
    run_final_audit()
