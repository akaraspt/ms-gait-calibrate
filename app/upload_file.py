import os

import PIL

from PIL import Image

from app import app
from config import UPLOAD_FOLDER, THUMBNAIL_FOLDER


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'zip', 'csv'])
IGNORED_FILES = set(['.gitignore', '.DS_Store'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def gen_file_name(filename):
    """
    If file was exist already, rename it and return a new name
    """

    i = 1
    while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        name, extension = os.path.splitext(filename)
        filename = '%s_%s%s' % (name, str(i), extension)
        i = i + 1

    return filename


def create_thumbnai(image):
    try:
        basewidth = 80
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], image))
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
        img.save(os.path.join(app.config['THUMBNAIL_FOLDER'], image))

        return True
    
    except:
        print traceback.format_exc()
        return False


class uploadfile():
    def __init__(self, name, type=None, size=None, not_allowed_msg=''):
        self.name = name
        self.type = type
        self.size = size
        self.not_allowed_msg = not_allowed_msg
        self.url = "{}/{}".format(UPLOAD_FOLDER, name)
        self.thumbnail_url = "{}/{}".format(THUMBNAIL_FOLDER, name)
        self.delete_url = "delete/{}".format(name)
        self.delete_type = "DELETE"


    def is_image(self):
        fileName, fileExtension = os.path.splitext(self.name.lower())

        if fileExtension in ['.jpg', '.png', '.jpeg', '.bmp']:
            return True

        return False


    def get_file(self):
        if self.type != None:
            # POST an image
            if self.type.startswith('image'):
                return {"name": self.name,
                        "type": self.type,
                        "size": self.size, 
                        "url": self.url, 
                        "thumbnailUrl": self.thumbnail_url,
                        "deleteUrl": self.delete_url, 
                        "deleteType": self.delete_type,}
            
            # POST an normal file
            elif self.not_allowed_msg == '':
                return {"name": self.name,
                        "type": self.type,
                        "size": self.size, 
                        "url": self.url, 
                        "deleteUrl": self.delete_url, 
                        "deleteType": self.delete_type,}

            # File type is not allowed
            else:
                return {"error": self.not_allowed_msg,
                        "name": self.name,
                        "type": self.type,
                        "size": self.size,}

        # GET image from disk
        elif self.is_image():
            return {"name": self.name,
                    "size": self.size, 
                    "url": self.url, 
                    "thumbnailUrl": self.thumbnail_url,
                    "deleteUrl": self.delete_url, 
                    "deleteType": self.delete_type,}
        
        # GET normal file from disk
        else:
            return {"name": self.name,
                    "size": self.size, 
                    "url": self.url, 
                    "deleteUrl": self.delete_url, 
                    "deleteType": self.delete_type,}
