from huggingface_hub import snapshot_download

MODEL_ID = "zhengchong/CatVTON"
SAVE_DIR = r"C:\Users\kavip\OneDrive\Documents\mini\CatVTON_model"

print("Downloading CatVTON...")
print("This downloads the CatVTON model repository, not the GitHub source code.")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_DIR,
    repo_type="model"
)

print("\nDownload completed!")
print("Saved to:")
print(SAVE_DIR)