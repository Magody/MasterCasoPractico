from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

class TTSProvider(ABC):
    @abstractmethod
    def generate(self, text: str, temp_path: Path = None) -> Union[bytes, Path]:
        """
        Generate speech audio for the given text.

        :param text: Text to synthesize.
        :param temp_path: Optional filesystem path where to write the WAV file.
                          If None, the method should return raw WAV bytes.
        :return: If temp_path is provided, return a Path to the file.
                 Otherwise, return a bytes object containing the WAV data.
        """
        raise NotImplementedError