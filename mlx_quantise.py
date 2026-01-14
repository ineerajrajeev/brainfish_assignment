import sys
from mlx_lm import convert
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL_ID = os.environ.get("MLX_MODEL_PATH")
OUTPUT_DIR = "mlx_model"

print(f"Downloading and Quantizing {MODEL_ID} to 4-bit MLX format...")
print("This may take a few minutes depending on your internet connection.")

try:
    # FIX: The argument name is 'hf_path', not 'repo'
    convert(
        hf_path=MODEL_ID,   # <--- Updated parameter name
        quantize=True,      # Enable 4-bit quantization
        upload_repo=None,   # Keep local, don't upload to HF
        output_path=OUTPUT_DIR
    )
    print(f"Success! Optimized model saved to: {OUTPUT_DIR}")
    print(f"You can now run your agent using this path.")
    
except Exception as e:
    print(f"Error during conversion: {e}")