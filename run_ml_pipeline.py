"""
Chạy toàn bộ pipeline ML một lần:
- Export data từ MySQL
- Tiền xử lý
- Backup model cũ (nếu có) và train lại model NMF

Yêu cầu:
- Đã cài deps: pip install -r requirements.txt
- Đã cấu hình .env (DB_*, OPENAI_* nếu cần)

Chạy:
    python run_ml_pipeline.py
"""

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n=== {name} ===")
    result = subprocess.run(cmd, cwd=BASE_DIR, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit {result.returncode})")


def main() -> None:
    # Dùng pipeline auto_retrain có sẵn (gồm export -> preprocess -> backup -> train)
    run_step(
        "Auto retrain pipeline",
        [sys.executable, "src/training/auto_retrain.py"],
    )
    print("\nHoàn tất. Model mới được lưu tại data/models/recommendation_model.pkl")
    print("Lưu ý: restart FastAPI recommendation service để load model mới.")


if __name__ == "__main__":
    main()

