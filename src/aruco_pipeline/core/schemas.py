from __future__ import annotations

from dataclasses import dataclass


def _require_keys(data: dict, required: list[str], source: str) -> None:
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} in {source}")


@dataclass
class FrameEntry:
    frame_index: int
    timestamp_s: float
    filename: str

    @classmethod
    def from_dict(cls, data: dict, *, source: str) -> FrameEntry:
        _require_keys(data, ["frame_index", "timestamp_s", "filename"], source)
        return cls(
            frame_index=data["frame_index"],
            timestamp_s=data["timestamp_s"],
            filename=data["filename"],
        )

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "filename": self.filename,
        }


@dataclass
class VideoMetadata:
    source_video: str
    fps: float
    total_frames: int
    resolution: dict[str, int]
    stride: int
    frames_extracted: int
    extracted_at: str
    opencv_version: str
    frames: list[FrameEntry]
    # Optional: 35mm-equivalent focal length probed from the video container
    # (None when metadata is absent, e.g. messaging-app transfers strip it).
    focal_length_35mm: float | None = None

    @classmethod
    def from_dict(cls, data: dict, *, source: str = "metadata.json") -> VideoMetadata:
        _require_keys(
            data,
            [
                "source_video",
                "fps",
                "total_frames",
                "resolution",
                "stride",
                "frames_extracted",
                "extracted_at",
                "opencv_version",
                "frames",
            ],
            source,
        )
        return cls(
            source_video=data["source_video"],
            fps=data["fps"],
            total_frames=data["total_frames"],
            resolution=data["resolution"],
            stride=data["stride"],
            frames_extracted=data["frames_extracted"],
            extracted_at=data["extracted_at"],
            opencv_version=data["opencv_version"],
            frames=[FrameEntry.from_dict(f, source=source) for f in data["frames"]],
            focal_length_35mm=data.get("focal_length_35mm"),
        )

    def to_dict(self) -> dict:
        data = {
            "source_video": self.source_video,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "resolution": self.resolution,
            "stride": self.stride,
            "frames_extracted": self.frames_extracted,
            "extracted_at": self.extracted_at,
            "opencv_version": self.opencv_version,
            "frames": [f.to_dict() for f in self.frames],
        }
        # Omitted when unset so old metadata files round-trip unchanged.
        if self.focal_length_35mm is not None:
            data["focal_length_35mm"] = self.focal_length_35mm
        return data


@dataclass
class DetectionEntry:
    filename: str
    frame_index: int
    markers_detected: int
    marker_ids: list[int]
    corners: list

    @classmethod
    def from_dict(cls, data: dict, *, source: str) -> DetectionEntry:
        _require_keys(
            data,
            ["filename", "frame_index", "markers_detected", "marker_ids", "corners"],
            source,
        )
        return cls(
            filename=data["filename"],
            frame_index=data["frame_index"],
            markers_detected=data["markers_detected"],
            marker_ids=data["marker_ids"],
            corners=data["corners"],
        )

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "frame_index": self.frame_index,
            "markers_detected": self.markers_detected,
            "marker_ids": self.marker_ids,
            "corners": self.corners,
        }


@dataclass
class DetectionsFile:
    dictionary: str
    total_frames: int
    frames_with_detections: int
    detections: list[DetectionEntry]

    @classmethod
    def from_dict(
        cls, data: dict, *, source: str = "detections.json"
    ) -> DetectionsFile:
        _require_keys(
            data,
            ["dictionary", "total_frames", "frames_with_detections", "detections"],
            source,
        )
        return cls(
            dictionary=data["dictionary"],
            total_frames=data["total_frames"],
            frames_with_detections=data["frames_with_detections"],
            detections=[
                DetectionEntry.from_dict(d, source=source) for d in data["detections"]
            ],
        )

    def to_dict(self) -> dict:
        return {
            "dictionary": self.dictionary,
            "total_frames": self.total_frames,
            "frames_with_detections": self.frames_with_detections,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class MarkerDetection:
    id: int
    corners: list

    @classmethod
    def from_dict(cls, data: dict, *, source: str) -> MarkerDetection:
        _require_keys(data, ["id", "corners"], source)
        return cls(id=data["id"], corners=data["corners"])

    def to_dict(self) -> dict:
        return {"id": self.id, "corners": self.corners}


@dataclass
class FilterManifest:
    dictionary: str
    min_markers: int
    total_frames_input: int
    frames_filtered_out: int
    frames: list[str]
    marker_detections: dict[str, list[MarkerDetection]]
    # Optional (added later): absent from manifests written by older versions.
    sharpness: dict | None = None
    tool_versions: dict | None = None
    camera: dict | None = None
    mask_dir: str | None = None
    mask_generation: dict | None = None

    @classmethod
    def from_dict(cls, data: dict, *, source: str = "manifest.json") -> FilterManifest:
        _require_keys(
            data,
            [
                "dictionary",
                "min_markers",
                "total_frames_input",
                "frames_filtered_out",
                "frames",
                "marker_detections",
            ],
            source,
        )
        return cls(
            dictionary=data["dictionary"],
            min_markers=data["min_markers"],
            total_frames_input=data["total_frames_input"],
            frames_filtered_out=data["frames_filtered_out"],
            frames=data["frames"],
            marker_detections={
                filename: [MarkerDetection.from_dict(m, source=source) for m in markers]
                for filename, markers in data["marker_detections"].items()
            },
            sharpness=data.get("sharpness"),
            tool_versions=data.get("tool_versions"),
            camera=data.get("camera"),
            mask_dir=data.get("mask_dir"),
            mask_generation=data.get("mask_generation"),
        )

    def to_dict(self) -> dict:
        data = {
            "dictionary": self.dictionary,
            "min_markers": self.min_markers,
            "total_frames_input": self.total_frames_input,
            "frames_filtered_out": self.frames_filtered_out,
            "frames": self.frames,
            "marker_detections": {
                filename: [m.to_dict() for m in markers]
                for filename, markers in self.marker_detections.items()
            },
        }
        # Optional fields are omitted when unset so old manifests round-trip.
        if self.sharpness is not None:
            data["sharpness"] = self.sharpness
        if self.tool_versions is not None:
            data["tool_versions"] = self.tool_versions
        if self.camera is not None:
            data["camera"] = self.camera
        if self.mask_dir is not None:
            data["mask_dir"] = self.mask_dir
        if self.mask_generation is not None:
            data["mask_generation"] = self.mask_generation
        return data


@dataclass
class MarkerSheetManifest:
    dictionary: str
    num_markers: int
    ids: list[int]
    side_pixels: int
    margin_pixels: int
    total_image_side_pixels: int
    dpi: int
    total_image_side_mm: float
    generated_at: str

    @classmethod
    def from_dict(
        cls, data: dict, *, source: str = "manifest.json"
    ) -> MarkerSheetManifest:
        _require_keys(
            data,
            [
                "dictionary",
                "num_markers",
                "ids",
                "side_pixels",
                "margin_pixels",
                "total_image_side_pixels",
                "dpi",
                "total_image_side_mm",
                "generated_at",
            ],
            source,
        )
        return cls(
            dictionary=data["dictionary"],
            num_markers=data["num_markers"],
            ids=data["ids"],
            side_pixels=data["side_pixels"],
            margin_pixels=data["margin_pixels"],
            total_image_side_pixels=data["total_image_side_pixels"],
            dpi=data["dpi"],
            total_image_side_mm=data["total_image_side_mm"],
            generated_at=data["generated_at"],
        )

    def to_dict(self) -> dict:
        return {
            "dictionary": self.dictionary,
            "num_markers": self.num_markers,
            "ids": self.ids,
            "side_pixels": self.side_pixels,
            "margin_pixels": self.margin_pixels,
            "total_image_side_pixels": self.total_image_side_pixels,
            "dpi": self.dpi,
            "total_image_side_mm": self.total_image_side_mm,
            "generated_at": self.generated_at,
        }


@dataclass
class ComparisonFile:
    dictionary: str
    model: str
    weights_path: str
    total_frames: int
    summary: dict
    frames: list[dict]

    @classmethod
    def from_dict(
        cls, data: dict, *, source: str = "comparison.json"
    ) -> ComparisonFile:
        _require_keys(
            data,
            [
                "dictionary",
                "model",
                "weights_path",
                "total_frames",
                "summary",
                "frames",
            ],
            source,
        )
        return cls(
            dictionary=data["dictionary"],
            model=data["model"],
            weights_path=data["weights_path"],
            total_frames=data["total_frames"],
            summary=data["summary"],
            frames=data["frames"],
        )

    def to_dict(self) -> dict:
        return {
            "dictionary": self.dictionary,
            "model": self.model,
            "weights_path": self.weights_path,
            "total_frames": self.total_frames,
            "summary": self.summary,
            "frames": self.frames,
        }
