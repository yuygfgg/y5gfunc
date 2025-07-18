from pathlib import Path
import sys
from typing import Union
from ..utils import resolve_path
from vstools import vs


class RpuFile:
    """
    Represents a Dolby Vision RPU binary file.
    """

    _START_CODE = b"\x00\x00\x00\x01"
    _RPU_PAYLOAD_START_BYTE = b"\x19"

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initializes the RpuFile instance.

        Args:
            file_path: Path to the RPU file.

        Raises:
            FileNotFoundError: If the file path does not exist.
        """
        self.file_path = resolve_path(file_path)
        self._data = b""
        self._nalu_offsets: list[int] = []

        self._load_and_parse()

    def _load_and_parse(self) -> None:
        with open(self.file_path, "rb") as f:
            self._data = f.read()

        current_pos = 0
        while True:
            offset = self._data.find(self._START_CODE, current_pos)
            if offset == -1:
                break
            self._nalu_offsets.append(offset)
            current_pos = offset + 1

        if not self._nalu_offsets:
            print(
                f"Warning: No RPU NALUs found in file '{self.file_path}'.",
                file=sys.stderr,
            )

    def __len__(self) -> int:
        return len(self._nalu_offsets)

    def get_frame_data(self, frame_index: int) -> bytes:
        """
        Gets the payload of a single RPU frame by its index.

        Args:
            frame_index: The 0-based index of the frame to retrieve.

        Returns:
            The RPU payload binary data (starting with 0x19) for the requested frame.

        Raises:
            IndexError: If frame_index is out of bounds.
            ValueError: If the RPU payload (0x19) is not found within the frame.
        """
        if not 0 <= frame_index < len(self._nalu_offsets):
            raise IndexError(
                f"Frame index {frame_index} is out of range. Valid indices are 0 to {len(self) - 1}."
            )

        start_offset = self._nalu_offsets[frame_index]

        if frame_index + 1 < len(self._nalu_offsets):
            end_offset = self._nalu_offsets[frame_index + 1]
        else:
            end_offset = len(self._data)

        frame_content_start = self._data.find(
            self._RPU_PAYLOAD_START_BYTE, start_offset, end_offset
        )

        if frame_content_start == -1:
            raise ValueError(
                f"Could not find RPU payload (0x19) in frame {frame_index}."
            )

        return self._data[frame_content_start:end_offset]


def write_rpu(rpu_file_path: Union[str, Path], clip: vs.VideoNode) -> vs.VideoNode:
    """
    Writes the RPU data to the clip.
    """
    rpu_file = RpuFile(rpu_file_path)

    def write_frame(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        rpu_data = rpu_file.get_frame_data(n)
        fout = f.copy()
        fout.props["DolbyVisionRPU"] = rpu_data
        return fout

    return clip.std.ModifyFrame(clip, selector=write_frame)
