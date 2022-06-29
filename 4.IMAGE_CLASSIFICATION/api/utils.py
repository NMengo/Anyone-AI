import os
import imghdr
import hashlib as hs

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(file):
    ext = file.filename.rsplit('.', 1)[1].lower()
    hs_file = hs.md5(file.read())
    new_file_name = hs_file.hexdigest()+'.'+ext
    # In order to be able to read the file again. Otherwise, it will stay in last point in memory.
    file.seek(0)
    return new_file_name