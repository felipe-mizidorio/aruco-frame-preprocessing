import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
from cv2.typing import MatLike
from fpdf import FPDF

from ..config import load_config
from ..core import pipeline_io
from ..core.schemas import MarkerSheetManifest

logger = logging.getLogger(__name__)

MM_PER_INCH = 25.4

PAGE_FORMATS_MM = {
    "A4": (210.0, 297.0),
    "Letter": (215.9, 279.4),
}


def generate_marker(dictionary: int, marker_id: int, side_pixels: int) -> MatLike:
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
    return cv2.aruco.generateImageMarker(aruco_dictionary, marker_id, side_pixels)


def add_white_margin(image: MatLike, margin_pixels: int) -> MatLike:
    return cv2.copyMakeBorder(
        image,
        margin_pixels,
        margin_pixels,
        margin_pixels,
        margin_pixels,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def pixels_to_mm(pixels: int, dpi: int) -> float:
    return pixels / dpi * MM_PER_INCH


def save_markers(
    num_markers: int,
    side_pixels: int,
    margin_pixels: int,
    output_dir: str | Path,
    dictionary_name: str,
    dpi: int,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary = pipeline_io.ARUCO_DICTIONARIES[dictionary_name]

    paths: list[Path] = []
    for i in range(num_markers):
        image = generate_marker(dictionary, i, side_pixels)
        image = add_white_margin(image, margin_pixels)
        out_path = output_dir / f"marker_{i}.png"
        cv2.imwrite(str(out_path), image)
        logger.info("Marker %d saved to %s", i, out_path)
        paths.append(out_path)

    total_side_pixels = side_pixels + 2 * margin_pixels
    manifest = MarkerSheetManifest(
        dictionary=dictionary_name,
        num_markers=num_markers,
        ids=list(range(num_markers)),
        side_pixels=side_pixels,
        margin_pixels=margin_pixels,
        total_image_side_pixels=total_side_pixels,
        dpi=dpi,
        total_image_side_mm=round(pixels_to_mm(total_side_pixels, dpi), 3),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = output_dir / "manifest.json"
    pipeline_io.save_json(manifest.to_dict(), manifest_path)
    logger.info("Manifest saved to %s", manifest_path)

    return paths


def build_marker_sheet(
    marker_paths: list[Path],
    marker_side_mm: float,
    output_path: str | Path,
    page_format: str = "A4",
    page_margin_mm: float = 15.0,
    gap_mm: float = 8.0,
    label_height_mm: float = 6.0,
) -> None:
    page_width_mm, page_height_mm = PAGE_FORMATS_MM[page_format]

    usable_width = page_width_mm - 2 * page_margin_mm
    usable_height = page_height_mm - 2 * page_margin_mm
    cell_width = marker_side_mm + gap_mm
    cell_height = marker_side_mm + label_height_mm + gap_mm

    cols = max(1, int(usable_width // cell_width))
    rows = max(1, int(usable_height // cell_height))
    per_page = cols * rows

    pdf = FPDF(unit="mm", format=page_format)
    pdf.set_font("helvetica", size=8)

    def draw_scale_bar() -> None:
        bar_x, bar_y, bar_len = page_margin_mm, 8.0, 10.0
        pdf.set_line_width(0.3)
        pdf.line(bar_x, bar_y, bar_x + bar_len, bar_y)
        pdf.set_xy(bar_x, bar_y + 1)
        pdf.cell(
            bar_len + 60,
            4,
            "10 mm scale bar - verify with a ruler after printing",
            align="L",
        )

    for index, marker_path in enumerate(marker_paths):
        position_in_page = index % per_page
        if position_in_page == 0:
            pdf.add_page()
            draw_scale_bar()

        col = position_in_page % cols
        row = position_in_page // cols
        x = page_margin_mm + col * cell_width
        y = page_margin_mm + row * cell_height

        pdf.image(str(marker_path), x=x, y=y, w=marker_side_mm, h=marker_side_mm)

        marker_id = marker_path.stem.split("_")[-1]
        pdf.set_xy(x, y + marker_side_mm + 0.5)
        pdf.cell(marker_side_mm, label_height_mm, f"ID {marker_id}", align="C")

    pdf.output(str(output_path))
    logger.info(
        "Marker sheet saved to %s (%d markers, %d per page)",
        output_path,
        len(marker_paths),
        per_page,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker images and a print-ready PDF sheet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-markers",
        type=int,
        default=None,
        help="Number of markers to generate (IDs 0 to N-1).",
    )
    parser.add_argument(
        "--side-pixels",
        type=int,
        default=None,
        help="Side length of the coded area, in pixels (excludes white margin).",
    )
    parser.add_argument(
        "--margin-pixels",
        type=int,
        default=None,
        help="White margin added around the marker, in pixels, on each side.",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default=None,
        choices=sorted(pipeline_io.ARUCO_DICTIONARIES.keys()),
        help="ArUco dictionary to use.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="DPI assumed when converting pixels to physical mm size for the PDF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/markers"),
        help="Directory to write marker images, manifest and PDF sheet.",
    )
    parser.add_argument(
        "--pdf-name",
        type=str,
        default="markers_sheet.pdf",
        help="Filename of the print-ready PDF sheet, saved inside output-dir.",
    )
    parser.add_argument(
        "--page-format",
        type=str,
        default=None,
        choices=sorted(PAGE_FORMATS_MM.keys()),
        help="Page format for the PDF sheet.",
    )
    return parser.parse_args()


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()

    cfg = load_config().markers
    num_markers = args.num_markers if args.num_markers is not None else cfg.num_markers
    side_pixels = args.side_pixels if args.side_pixels is not None else cfg.side_pixels
    margin_pixels = (
        args.margin_pixels if args.margin_pixels is not None else cfg.margin_pixels
    )
    dictionary = args.dictionary if args.dictionary is not None else cfg.dictionary
    dpi = args.dpi if args.dpi is not None else cfg.dpi
    page_format = args.page_format if args.page_format is not None else cfg.page_format

    marker_paths = save_markers(
        num_markers,
        side_pixels,
        margin_pixels,
        args.output_dir,
        dictionary,
        dpi,
    )

    total_side_pixels = side_pixels + 2 * margin_pixels
    marker_side_mm = pixels_to_mm(total_side_pixels, dpi)

    build_marker_sheet(
        marker_paths,
        marker_side_mm,
        Path(args.output_dir) / args.pdf_name,
        page_format=page_format,
    )


if __name__ == "__main__":
    main()
