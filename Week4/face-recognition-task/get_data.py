import sys

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


import zipfile

def unpack(filename):
    with zipfile.ZipFile(filename) as zf:
        zf.extractall()

