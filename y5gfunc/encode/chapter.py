from typing import Union, Optional, Any
from pathlib import Path
import subprocess
import struct
from ..utils import resolve_path


def get_bd_chapter(
    m2ts_or_mpls_path: Union[str, Path],
    chapter_save_path: Union[str, Path],
    target_clip: Optional[str] = None,
    all: bool = False,  # True: return all mpls marks; False: return chapter
) -> Path:
    """
    Extracts chapters from a Blu-ray MPLS or M2TS file to an OGM chapter file.

    If an M2TS path is provided, it searches the corresponding BDMV structure for the MPLS file containing that M2TS clip.

    Args:
        m2ts_or_mpls_path: Path to the input M2TS or MPLS file.
        chapter_save_path: Path to save the output OGM chapter file.
        target_clip: The 5-digit clip name (e.g., "00001") if input is MPLS and `all` is False. Auto-detected if input is M2TS.
        all: If True, extract all chapter marks from the relevant MPLS. If False, extract chapters relative to the target_clip's start time.

    Returns:
        Path to the saved OGM chapter file.

    Raises:
        FileNotFoundError: If input path or required BDMV structure doesn't exist.
        ValueError: If input is invalid (e.g., MPLS without target_clip/all=True, or no chapters found).
        RuntimeError: On errors during MPLS parsing.
        IOError: On failure to write the chapter file.
    """

    m2ts_or_mpls_path = resolve_path(m2ts_or_mpls_path)
    chapter_save_path = resolve_path(chapter_save_path)

    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        whole_seconds = int(seconds_remainder)
        milliseconds = int((seconds_remainder - whole_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

    def _process_mpls(
        mpls_path: Path, target_clip: Optional[str] = None
    ) -> Optional[list[float]]:
        try:
            with mpls_path.open("rb") as f:
                if f.read(4) != b"MPLS":
                    raise ValueError(
                        f"get_bd_chapter: Invalid MPLS format in file: {mpls_path}"
                    )

                f.seek(0)
                header: dict[str, Any] = {}
                header["TypeIndicator"] = f.read(4)
                header["VersionNumber"] = f.read(4)
                (header["PlayListStartAddress"],) = struct.unpack(">I", f.read(4))
                (header["PlayListMarkStartAddress"],) = struct.unpack(">I", f.read(4))
                (header["ExtensionDataStartAddress"],) = struct.unpack(">I", f.read(4))

                f.seek(header["PlayListStartAddress"])
                (playlist_length,) = struct.unpack(">I", f.read(4))
                f.read(2)  # reserved
                (num_items,) = struct.unpack(">H", f.read(2))
                (num_subpaths,) = struct.unpack(">H", f.read(2))

                play_items = []
                target_item_index = None
                for i in range(num_items):
                    (item_length,) = struct.unpack(">H", f.read(2))
                    item_start = f.tell()

                    clip_name = f.read(5).decode("utf-8", errors="ignore")
                    codec_id = f.read(4).decode("utf-8", errors="ignore")  # noqa: F841

                    f.read(3)  # reserved
                    stc_id = f.read(1)  # noqa: F841
                    (in_time,) = struct.unpack(">I", f.read(4))
                    (out_time,) = struct.unpack(">I", f.read(4))

                    if target_clip and clip_name == target_clip:
                        target_item_index = i

                    play_items.append(
                        {
                            "clip_name": clip_name,
                            "in_time": in_time,
                            "out_time": out_time,
                        }
                    )

                    f.seek(item_start + item_length)

                if target_clip and target_item_index is None:
                    return None

                f.seek(header["PlayListMarkStartAddress"])
                (marks_length,) = struct.unpack(">I", f.read(4))
                (num_marks,) = struct.unpack(">H", f.read(2))

                chapters_by_item = {}
                for _ in range(num_marks):
                    f.read(1)  # reserved
                    (mark_type,) = struct.unpack(">B", f.read(1))
                    (ref_play_item_id,) = struct.unpack(">H", f.read(2))
                    (mark_timestamp,) = struct.unpack(">I", f.read(4))
                    (entry_es_pid,) = struct.unpack(">H", f.read(2))
                    (duration,) = struct.unpack(">I", f.read(4))

                    if mark_type == 1:
                        if ref_play_item_id not in chapters_by_item:
                            chapters_by_item[ref_play_item_id] = []
                        chapters_by_item[ref_play_item_id].append(mark_timestamp)

                result = []
                if target_clip:
                    if target_item_index in chapters_by_item:
                        marks = chapters_by_item[target_item_index]
                        offset = min(marks)
                        if play_items[target_item_index]["in_time"] < offset:  # type: ignore[index]
                            offset = play_items[target_item_index]["in_time"]  # type: ignore[index]

                        for timestamp in marks:
                            relative_time = (timestamp - offset) / 45000.0
                            if relative_time >= 0:
                                result.append(relative_time)
                else:
                    for item_id, marks in chapters_by_item.items():
                        offset = min(marks)
                        if play_items[item_id]["in_time"] < offset:
                            offset = play_items[item_id]["in_time"]

                        for timestamp in marks:
                            relative_time = (timestamp - offset) / 45000.0
                            if relative_time >= 0:
                                result.append(relative_time)

                return sorted(result)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"get_bd_chapter: Error processing MPLS file: {str(e)}")

    if not m2ts_or_mpls_path.exists():
        raise FileNotFoundError(
            f"get_bd_chapter: Path does not exist: {m2ts_or_mpls_path}"
        )

    is_mpls = m2ts_or_mpls_path.suffix.lower() == ".mpls"

    if is_mpls:
        if not target_clip and not all:
            raise ValueError(
                "get_bd_chapter: target_clip must be provided with MPLS input if all is False!"
            )
        chapters = (
            _process_mpls(m2ts_or_mpls_path, target_clip)
            if not all
            else _process_mpls(m2ts_or_mpls_path)
        )
    else:
        bdmv_root = next(
            (p.parent for p in m2ts_or_mpls_path.parents if p.name.upper() == "BDMV"),
            None,
        )
        if not bdmv_root:
            raise FileNotFoundError(
                "get_bd_chapter: Could not find BDMV directory in path hierarchy"
            )

        target_clip = m2ts_or_mpls_path.stem
        mpls_dir = bdmv_root / "BDMV" / "PLAYLIST"

        if not mpls_dir.exists():
            raise FileNotFoundError(f"PLAYLIST directory not found: {mpls_dir}")

        chapters = None
        for mpls_file in mpls_dir.glob("*.mpls"):
            try:
                chapters = _process_mpls(mpls_file, target_clip=target_clip)
                if chapters:
                    if all:
                        chapters = _process_mpls(mpls_file)
                    break
            except (ValueError, RuntimeError):
                continue

    if not chapters:
        raise ValueError("get_bd_chapter: No chapters found in the Blu-ray disc")

    try:
        with chapter_save_path.open("w", encoding="utf-8") as f:
            for i, time in enumerate(chapters, 1):
                chapter_num = f"{i:02d}"
                timestamp = _format_timestamp(time)
                f.write(f"CHAPTER{chapter_num}={timestamp}\n")
                f.write(f"CHAPTER{chapter_num}NAME=Chapter {i}\n")
    except IOError as e:
        raise IOError(f"get_bd_chapter: Failed to write chapter file: {str(e)}")

    return chapter_save_path


def get_mkv_chapter(mkv_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    """
    Extracts chapters from an MKV file using mkvextract to an OGM chapter file.

    Args:
        mkv_path: Path to the input MKV file.
        output_path: Path to save the output OGM chapter file.

    Returns:
        Path to the saved OGM chapter file.

    Raises:
        RuntimeError: If mkvextract fails to extract chapters.
        IOError: On failure to write the chapter file.
    """
    mkv_path = resolve_path(mkv_path)
    output_path = resolve_path(output_path)

    try:
        result = subprocess.run(
            ["mkvextract", "chapters", str(mkv_path), "-s"],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"get_mkv_chapter: Error extracting chapters from '{mkv_path}': {e.stderr.strip()}"
        )

    chapter_data = result.stdout

    try:
        output_path.write_text(chapter_data, encoding="utf-8")
    except IOError as e:
        raise IOError(
            f"get_mkv_chapter: Failed to write chapter file '{output_path}': {str(e)}"
        )

    return output_path
