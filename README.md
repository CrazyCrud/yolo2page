# YOLO to PAGE-XML Converter

Convert YOLO segmentation format annotations to PAGE-XML format for document layout analysis.

## Features

- Converts YOLO polygon segmentations to PAGE-XML regions
- Supports nested TextRegions (articles → paragraphs → lines)
- Automatic parent-child assignment based on geometric overlap
- Handles multiple region types (TextRegion, TableRegion, ImageRegion, etc.)
- Smart orphan handling for unassigned elements

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command Line

```bash
# Convert folders
python converter.py labels_folder/ images_folder/ --mapping mapping.json --output pagexml/

# Convert single file
python converter.py label.txt image.jpg --mapping mapping.json --output page.xml
```

### Python API

```python
from converter import YoloPageConverter

# Initialize with mapping
converter = YoloPageConverter(mapping_file="mapping.json")

# Convert single file
converter.convert_page("labels/page001.txt", "images/page001.jpg", "output/page001.xml")

# Convert folder
converter.convert_folder("labels_folder", "images_folder", "pagexml_output")
```

## Mapping Configuration

Create a JSON file to map YOLO class IDs to PAGE-XML elements:

```json
{
    "0": {
        "element": "TextRegion",
        "type": "article"
    },
    "1": {
        "element": "TextRegion",
        "type": "paragraph",
        "parent": "TextRegion"
    },
    "2": {
        "element": "TextLine",
        "parent": "TextRegion",
        "parent_type": "paragraph"
    },
    "3": {
        "element": "TableRegion"
    },
    "4": {
        "element": "ImageRegion"
    }
}
```

### Key Mapping Properties

- `element`: PAGE-XML element type (required)
- `type`: Region subtype for TextRegions (optional)
- `parent`: Parent element type for nesting (optional)
- `parent_type`: Default type for orphan regions (TextLines only)

## How It Works

1. **Parent TextRegions** created first (articles, columns)
2. **Child TextRegions** nested based on 50%+ containment
3. **TextLines** assigned to smallest overlapping region
4. **Orphans** get new regions if no overlap found

## Supported Elements

- `TextRegion` (with types: heading, paragraph, caption, footer, etc.)
- `TextLine`
- `TableRegion`
- `ImageRegion`
- `GraphicRegion`
- `SeparatorRegion`
- `NoiseRegion`

## Input Format

- **Labels**: YOLO format text files with polygon segmentation
  ```
  class_id x1 y1 x2 y2 x3 y3 ...
  ```
- **Images**: Matching image files (same base name)
- **Coordinates**: Normalized [0,1] YOLO format

## Output

Valid PAGE-XML 2019-07-15 format with:
- Proper hierarchical structure
- Absolute pixel coordinates
- Region types and relationships
- Metadata and schema compliance

## Example Structure

```xml
<TextRegion id="r0000" type="article">
    <Coords points="..."/>
    <TextRegion id="r0001" type="paragraph">
        <Coords points="..."/>
        <TextLine id="l0001">
            <Coords points="..."/>
        </TextLine>
    </TextRegion>
</TextRegion>
```

## Notes

- Hierarchy inferred from geometric overlap (YOLO is not hierarchical)
- TextLines require parent TextRegions per PAGE-XML spec
- Supports multiple nesting levels for complex layouts
