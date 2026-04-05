from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from manuscript import Pipeline
from manuscript.utils import visualize_page


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "collages"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "example_images_collage.png"
BACKGROUND = (248, 248, 245)
PANEL_BACKGROUND = (255, 255, 255)
TEXT_COLOR = (28, 28, 28)
BORDER_COLOR = (220, 220, 220)
RESAMPLE_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)
GRID_COLUMNS = 2
TITLE_FONT_SIZE = 18
TEXT_FONT_SIZE = 10
PADDING = 10
CARD_PADDING = 10
CARD_GAP = 8
TITLE_PANEL_PADDING = 6
TITLE_TOP_PADDING = 6
TITLE_CONTENT_GAP = 6
IMAGE_CELL_WIDTH = 380
IMAGE_CELL_HEIGHT = 300
TEXT_PANEL_WIDTH = 180
TEXT_LINE_SPACING = 1
SHOW_ORDER = True
OUTPUT_WIDTH = 2048
COLLAGE_ITEMS = [
    (0, "Рукописный дореформенный"),
    (1, "Печатный дореформенный"),
    (2, "Современный русский"),
    (4, "Русский текст на уличной сцене"),
]


def iter_images(input_dir: Path) -> list[Path]:
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return image_paths


def load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def normalize_preview_text(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return "Text was not recognized."
    return normalized


def _measure_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text or " ", font=font)
    return bbox[2] - bbox[0]


def split_long_token(
    token: str,
    max_width: int,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
) -> Iterable[str]:
    chunk = ""
    for char in token:
        candidate = chunk + char
        if chunk and _measure_text(candidate, draw, font) > max_width:
            yield chunk
            chunk = char
        else:
            chunk = candidate
    if chunk:
        yield chunk


def wrap_text(
    text: str,
    max_width: int,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
) -> list[str]:
    wrapped_lines: list[str] = []
    paragraphs = text.splitlines() or [text]

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line = ""
        for word in words:
            word_parts = list(split_long_token(word, max_width, draw, font))
            for part in word_parts:
                candidate = part if not current_line else f"{current_line} {part}"
                if _measure_text(candidate, draw, font) <= max_width:
                    current_line = candidate
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = part

        if current_line:
            wrapped_lines.append(current_line)

    return wrapped_lines or [""]


def ellipsize_to_width(
    text: str,
    max_width: int,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
) -> str:
    if _measure_text(text, draw, font) <= max_width:
        return text

    ellipsis = "..."
    trimmed = text.rstrip()
    while trimmed and _measure_text(trimmed + ellipsis, draw, font) > max_width:
        trimmed = trimmed[:-1].rstrip()
    return (trimmed + ellipsis) if trimmed else ellipsis


def render_text_panel(
    text: str,
    panel_width: int,
    font: ImageFont.ImageFont,
    padding: int,
    min_height: int = 140,
    line_spacing: int | None = None,
    fixed_height: int | None = None,
) -> Image.Image:
    measure_image = Image.new("RGB", (panel_width, 10), PANEL_BACKGROUND)
    measure_draw = ImageDraw.Draw(measure_image)
    content_width = max(120, panel_width - padding * 2)
    wrapped_lines = wrap_text(text, content_width, measure_draw, font)

    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
    else:
        line_height = max(getattr(font, "size", 16), 16)
    if line_spacing is None:
        line_spacing = max(8, line_height // 4)
    if fixed_height is not None:
        panel_height = fixed_height
        available_height = max(0, panel_height - padding * 2)
        max_lines = max(1, (available_height + line_spacing) // (line_height + line_spacing))
        wrapped_lines = wrapped_lines[:max_lines]
        if wrapped_lines and len(wrapped_lines) < len(wrap_text(text, content_width, measure_draw, font)):
            wrapped_lines[-1] = ellipsize_to_width(
                wrapped_lines[-1],
                content_width,
                measure_draw,
                font,
            )
    else:
        text_height = len(wrapped_lines) * line_height + max(0, len(wrapped_lines) - 1) * line_spacing
        panel_height = max(padding * 2 + text_height, min_height)

    panel = Image.new("RGB", (panel_width, panel_height), PANEL_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    y = padding
    for line in wrapped_lines:
        draw.text((padding, y), line, font=font, fill=TEXT_COLOR)
        y += line_height + line_spacing

    draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline=BORDER_COLOR, width=2)
    return panel


def render_title_panel(
    text: str,
    panel_width: int,
    font: ImageFont.ImageFont,
    padding: int,
) -> Image.Image:
    return render_text_panel(
        text=text,
        panel_width=panel_width,
        font=font,
        padding=padding,
        min_height=46,
        line_spacing=0,
    )


def add_image_border(image: Image.Image, border: int) -> Image.Image:
    framed = Image.new(
        "RGB",
        (image.width + border * 2, image.height + border * 2),
        PANEL_BACKGROUND,
    )
    framed.paste(image, (border, border))
    draw = ImageDraw.Draw(framed)
    draw.rectangle((0, 0, framed.width - 1, framed.height - 1), outline=BORDER_COLOR, width=2)
    return framed


def fit_image_to_box(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    resized_size = (
        max(1, int(image.width * scale)),
        max(1, int(image.height * scale)),
    )
    resized = image.resize(resized_size, RESAMPLE_LANCZOS)

    canvas = Image.new("RGB", (width, height), PANEL_BACKGROUND)
    offset_x = (width - resized.width) // 2
    offset_y = (height - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def resize_output(image: Image.Image, target_width: int) -> Image.Image:
    if image.width == target_width:
        return image
    target_height = max(1, int(image.height * target_width / image.width))
    return image.resize((target_width, target_height), RESAMPLE_LANCZOS)


def build_visualization(
    pipeline: Pipeline,
    image_path: Path,
    show_order: bool,
) -> tuple[Image.Image, str]:
    result = pipeline.predict(image_path)
    page = result["page"]
    vis_image = visualize_page(
        image_path,
        page,
        show_order=show_order,
        show_lines=False,
        show_numbers=False,
    ).convert("RGB")
    text = pipeline.get_text(page)
    return vis_image, text


def create_grid_collage(
    cards: list[tuple[str, Image.Image, str]],
    title_font: ImageFont.ImageFont,
    text_font: ImageFont.ImageFont,
    padding: int,
) -> Image.Image:
    card_panels: list[Image.Image] = []

    for title, image, text in cards:
        image_panel = add_image_border(
            fit_image_to_box(image, IMAGE_CELL_WIDTH, IMAGE_CELL_HEIGHT),
            border=6,
        )
        text_panel = render_text_panel(
            normalize_preview_text(text),
            panel_width=TEXT_PANEL_WIDTH,
            font=text_font,
            padding=CARD_PADDING,
            line_spacing=TEXT_LINE_SPACING,
            fixed_height=image_panel.height,
        )
        content_height = max(image_panel.height, text_panel.height)
        content_width = image_panel.width + CARD_GAP + text_panel.width
        title_panel = render_title_panel(
            text=title,
            panel_width=content_width,
            font=title_font,
            padding=TITLE_PANEL_PADDING,
        )

        card_width = content_width + CARD_PADDING * 2
        card_height = (
            TITLE_TOP_PADDING
            + title_panel.height
            + TITLE_CONTENT_GAP
            + content_height
            + CARD_PADDING
        )
        card = Image.new("RGB", (card_width, card_height), PANEL_BACKGROUND)

        title_x = (card_width - title_panel.width) // 2
        title_y = TITLE_TOP_PADDING
        card.paste(title_panel, (title_x, title_y))

        content_top = title_y + title_panel.height + TITLE_CONTENT_GAP
        image_y = content_top
        text_y = content_top
        card.paste(image_panel, (CARD_PADDING, image_y))
        card.paste(text_panel, (CARD_PADDING + image_panel.width + CARD_GAP, text_y))

        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, card_width - 1, card_height - 1), outline=BORDER_COLOR, width=2)
        card_panels.append(card)

    card_width = max(panel.width for panel in card_panels)
    card_height = max(panel.height for panel in card_panels)
    row_count = (len(card_panels) + GRID_COLUMNS - 1) // GRID_COLUMNS
    canvas_width = GRID_COLUMNS * card_width + (GRID_COLUMNS + 1) * padding
    canvas_height = row_count * card_height + (row_count + 1) * padding
    collage = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)

    for index, card_panel in enumerate(card_panels):
        row = index // GRID_COLUMNS
        col = index % GRID_COLUMNS
        x = padding + col * (card_width + padding)
        y = padding + row * (card_height + padding)
        collage.paste(card_panel, (x, y))

    return collage


def main() -> None:
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    image_paths = iter_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")
    card_count = len(COLLAGE_ITEMS)
    required_image_count = max(index for index, _ in COLLAGE_ITEMS) + 1
    if len(image_paths) < required_image_count:
        raise ValueError(
            f"Expected at least {required_image_count} images in {input_dir}, got {len(image_paths)}."
        )

    pipeline = Pipeline()
    title_font = load_font(TITLE_FONT_SIZE)
    text_font = load_font(TEXT_FONT_SIZE)
    cards: list[tuple[str, Image.Image, str]] = []

    for image_index, title in COLLAGE_ITEMS:
        image_path = image_paths[image_index]
        vis_image, text = build_visualization(
            pipeline=pipeline,
            image_path=image_path,
            show_order=SHOW_ORDER,
        )
        cards.append((title, vis_image, text))

    collage = create_grid_collage(
        cards=cards,
        title_font=title_font,
        text_font=text_font,
        padding=PADDING,
    )
    collage = resize_output(collage, OUTPUT_WIDTH)
    collage.save(DEFAULT_OUTPUT_FILE)
    print(DEFAULT_OUTPUT_FILE)


if __name__ == "__main__":
    main()
