import os

from app import app
from config import (UPLOAD_FOLDER,
                    THUMBNAIL_FOLDER,
                    TMP_FOLDER,
                    CALIBRATE_FOLDER,
                    MODEL_FOLDER,
                    OUTPUT_FOLDER)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(THUMBNAIL_FOLDER):
        os.makedirs(THUMBNAIL_FOLDER)
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
    if not os.path.exists(CALIBRATE_FOLDER):
        os.makedirs(CALIBRATE_FOLDER)
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    app.run(
        host="0.0.0.0", 
        threaded=True
    )