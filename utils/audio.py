import torchaudio


def _is_stereo(file_path) -> bool:
    """
    Return true if the file has more than one channel
    @param file_path:
    @return: bool
    """
    return torchaudio.info(file_path).num_channels > 1

