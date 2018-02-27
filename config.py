# Enable Flask's debugging features. Should be False in production
DEBUG = True

# Upload files
SECRET_KEY = 'p@ssw0rd'
UPLOAD_FOLDER = 'data'
THUMBNAIL_FOLDER = 'data/thumbnail'
TMP_FOLDER = 'data/tmp'
CALIBRATE_FOLDER = 'data/calibrate'
MODEL_FOLDER = 'data/model'
OUTPUT_FOLDER = 'data/output'
MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024