import imagehash
import hashlib

def get_phash(image):
    """
    :param image: PIL image
    :return: phash of the image
    """
    return imagehash.phash(image)

def get_md5(image):
    """
    :param image: PIL image
    :return: md5 of the iamge
    """
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()