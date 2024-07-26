import os
import re
import unicodedata


def generate_secure_filename(filename: str) -> str:
    r"""
    Adapted from werkzeug.utils import secure_filename

    Pass it a filename and it will return a secure version of it.  This
    filename can then safely be stored on a regular file system and passed
    to :func:`os.path.join`.  The filename returned is an ASCII only string
    for maximum portability.

    On windows systems the function also makes sure that the file is not
    named after one of the special device files.

    >>> generate_secure_filename("My cool movie.mov")
    'My_cool_movie.mov'
    >>> generate_secure_filename("../../../etc/passwd")
    'etc_passwd'
    >>> generate_secure_filename('i contain cool \xfcml\xe4uts.txt')
    'i_contain_cool_umlauts.txt'

    The function might return an empty filename.  It's your responsibility
    to ensure that the filename is unique and that you abort or
    generate a random filename if the function returned an empty one.

    .. versionadded:: 0.5

    :param filename: the filename to secure
    """
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    for sep in os.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, " ")
    filename = str(re.compile(r"[^A-Za-z0-9_.-]").sub("", "_".join(filename.split()))).strip("._")

    return filename
