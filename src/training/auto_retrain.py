"""
Auto retrain model (manual or schedule via Task Scheduler / Cron).
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

# Base dir of ML project (CodeSphere_ML)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Setup logging (store logs in CodeSphere_ML/logs)
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'retrain_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)

def retrain_model():
    """Automatic retrain model"""
    base_dir = BASE_DIR
    
    try:
        logging.info("=" * 50)
        logging.info("BAT DAU RETRAIN MODEL")
        logging.info(f"Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 50)
        
        # 1. Export data
        logging.info("Buoc 1: Export data tu database...")
        result = subprocess.run(
            [sys.executable, "src/data_collection/export_data.py"],
            cwd=base_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Export data failed: {result.stderr}")
        logging.info("Export data done")
        
        # 2. Preprocess
        logging.info("Buoc 2: Tien xu ly du lieu...")
        result = subprocess.run(
            [sys.executable, "src/preprocessing/preprocess.py"],
            cwd=base_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Preprocess failed: {result.stderr}")
        logging.info("Preprocess done")
        
        # 3. Backup old model (if exists)
        model_file = os.path.join(base_dir, "data/models/recommendation_model.pkl")
        if os.path.exists(model_file):
            backup_dir = os.path.join(base_dir, "data/models/backups")
            os.makedirs(backup_dir, exist_ok=True)
            backup_file = os.path.join(
                backup_dir, 
                f"recommendation_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            import shutil
            shutil.copy2(model_file, backup_file)
            logging.info(f"Da backup model cu: {backup_file}")
        
        # 4. Train model
        logging.info("Buoc 3: Train model moi...")
        result = subprocess.run(
            [sys.executable, "src/training/train_recommendation_model_simple.py"],
            cwd=base_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Train model failed: {result.stderr}")
        logging.info("Train model done")
        
        logging.info("=" * 50)
        logging.info("RETRAIN HOAN TAT!")
        logging.info("=" * 50)
        logging.info("Luu y: Can restart Python API de load model moi")
        
        # TODO: auto restart API or send signal to reload
        
    except Exception as e:
        logging.error(f"LOI khi retrain: {e}")
        logging.error(f"Error details: {str(e)}")
        raise

if __name__ == "__main__":
    retrain_model()

