import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Optional, Set
import uuid
from datetime import datetime
import logging
from shapely.geometry import Polygon
from shapely.ops import unary_union


class YoloPageConverter:
    def __init__(self, mapping_file: str = None):
        """
        Initialize the YOLO to PAGE-XML converter.

        Args:
            mapping_file: Path to JSON file containing YOLO to PAGE-XML element mappings
        """
        self.PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

        # Register namespace
        ET.register_namespace('', self.PAGE_NS)

        # Setup logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load mapping if provided, otherwise use default
        if mapping_file and Path(mapping_file).exists():
            self.load_mapping(mapping_file)
        else:
            self.create_default_mapping()
            if mapping_file:
                self.logger.warning(f"Mapping file not found: {mapping_file}. Using defaults.")

    def create_default_mapping(self):
        """Create a default mapping configuration."""
        self.mapping = {
            "0": {
                "element": "TextLine",
                "parent": "TextRegion",
                "parent_type": "paragraph"
            }
        }
        self.logger.info("Using default mapping for TextLine elements")

    def load_mapping(self, mapping_file: str):
        """
        Load element mapping from JSON file.

        Args:
            mapping_file: Path to JSON mapping file
        """
        with open(mapping_file, 'r') as f:
            self.mapping = json.load(f)

        self.logger.info(f"Loaded mapping from {mapping_file}")
        for class_id, config in self.mapping.items():
            if isinstance(config, dict):
                element = config.get('element', 'Unknown')
                parent = config.get('parent', 'none')
                self.logger.info(f"  Class {class_id}: {element} (parent: {parent})")

    def _parse_yolo_label(self, label_line: str) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Parse a YOLO segmentation label line.

        Args:
            label_line: YOLO format line "class_id x1 y1 x2 y2 ..."

        Returns:
            Tuple of (class_id, normalized_points)
        """
        parts = label_line.strip().split()
        if len(parts) < 7:  # class_id + at least 3 points (6 coordinates)
            return None, []

        class_id = int(parts[0])

        # Parse normalized coordinates
        points = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                x = float(parts[i])
                y = float(parts[i + 1])
                points.append((x, y))

        return class_id, points

    def _denormalize_points(self, points: List[Tuple[float, float]],
                            img_width: int, img_height: int) -> List[Tuple[int, int]]:
        """
        Convert normalized YOLO coordinates to absolute pixel coordinates.

        Args:
            points: List of normalized (x, y) tuples
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List of absolute (x, y) pixel coordinates
        """
        denormalized = []
        for x, y in points:
            abs_x = int(x * img_width)
            abs_y = int(y * img_height)
            denormalized.append((abs_x, abs_y))
        return denormalized

    def _points_to_string(self, points: List[Tuple[int, int]]) -> str:
        """
        Convert list of points to PAGE-XML points string format.

        Args:
            points: List of (x, y) tuples

        Returns:
            Space-separated string of comma-separated coordinates
        """
        return ' '.join([f'{x},{y}' for x, y in points])

    def _calculate_polygon_overlap(self, poly1_points: List[Tuple[int, int]],
                                   poly2_points: List[Tuple[int, int]]) -> float:
        """
        Calculate the overlap area between two polygons.

        Args:
            poly1_points: Points of the first polygon (e.g., TextLine)
            poly2_points: Points of the second polygon (e.g., TextRegion)

        Returns:
            Overlap area in pixels squared
        """
        try:
            # Create Shapely polygons
            poly1 = Polygon(poly1_points)
            poly2 = Polygon(poly2_points)

            # Check if polygons are valid
            if not poly1.is_valid or not poly2.is_valid:
                return 0.0

            # Calculate intersection
            intersection = poly1.intersection(poly2)

            # Return intersection area
            return intersection.area
        except Exception as e:
            self.logger.debug(f"Error calculating overlap: {e}")
            return 0.0

    def _calculate_containment_score(self, child_points: List[Tuple[int, int]],
                                     parent_points: List[Tuple[int, int]]) -> float:
        """
        Calculate how much of the child polygon is contained within the parent.

        Args:
            child_points: Points of the child polygon
            parent_points: Points of the parent polygon

        Returns:
            Percentage of child area contained in parent (0.0 to 1.0)
        """
        try:
            child_poly = Polygon(child_points)
            parent_poly = Polygon(parent_points)

            if not child_poly.is_valid or not parent_poly.is_valid:
                return 0.0

            intersection = child_poly.intersection(parent_poly)

            if child_poly.area > 0:
                return intersection.area / child_poly.area
            return 0.0
        except Exception as e:
            self.logger.debug(f"Error calculating containment: {e}")
            return 0.0

    def _find_best_parent(self, child_points: List[Tuple[int, int]],
                          potential_parents: List[Dict],
                          threshold: float = 0.5) -> Optional[ET.Element]:
        """
        Find the best parent region for a child based on containment.

        Args:
            child_points: Points of the child element
            potential_parents: List of potential parent regions
            threshold: Minimum containment score to consider (0.0 to 1.0)

        Returns:
            Best parent element or None
        """
        best_parent = None
        best_score = threshold  # Minimum threshold

        for parent_info in potential_parents:
            parent_element = parent_info['element']
            parent_points = parent_info['points']

            # Calculate how much of the child is contained in this parent
            score = self._calculate_containment_score(child_points, parent_points)

            if score > best_score:
                best_score = score
                best_parent = parent_element

        if best_parent and best_score > threshold:
            self.logger.debug(f"Found parent with {best_score:.1%} containment")
            return best_parent
        return None

    def _create_page_element(self, img_path: Path, img_width: int, img_height: int) -> Tuple[ET.Element, ET.Element]:
        """
        Create the root Page element with metadata.

        Args:
            img_path: Path to the image file
            img_width: Image width
            img_height: Image height

        Returns:
            Tuple of (root, page) XML elements
        """
        # Create root PcGts element
        root = ET.Element('PcGts')
        root.set('xmlns', self.PAGE_NS)
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:schemaLocation',
                 f'{self.PAGE_NS} http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd')

        # Add metadata
        metadata = ET.SubElement(root, 'Metadata')
        creator = ET.SubElement(metadata, 'Creator')
        creator.text = 'YoloPageConverter'
        created = ET.SubElement(metadata, 'Created')
        created.text = datetime.now().isoformat()
        last_change = ET.SubElement(metadata, 'LastChange')
        last_change.text = datetime.now().isoformat()

        # Add Page element
        page = ET.SubElement(root, 'Page')
        page.set('imageFilename', str(img_path.name))
        page.set('imageWidth', str(img_width))
        page.set('imageHeight', str(img_height))

        return root, page

    def _add_region(self, parent: ET.Element, element_type: str, points: List[Tuple[int, int]],
                    region_id: str = None, region_type: str = None) -> ET.Element:
        """
        Add a region element to a parent (Page or another Region).

        Args:
            parent: Parent element (Page or TextRegion)
            element_type: Type of region element
            points: List of absolute coordinate points
            region_id: Optional region ID
            region_type: Optional region type attribute

        Returns:
            Created region element
        """
        if region_id is None:
            prefix = element_type.replace('Region', '').lower()
            region_id = f"{prefix}_{uuid.uuid4().hex[:8]}"

        region = ET.SubElement(parent, element_type)
        region.set('id', region_id)

        if region_type and element_type == 'TextRegion':
            region.set('type', region_type)

        # Add Coords
        coords = ET.SubElement(region, 'Coords')
        coords.set('points', self._points_to_string(points))

        return region

    def _add_text_line(self, parent: ET.Element, points: List[Tuple[int, int]],
                       line_id: str = None) -> ET.Element:
        """
        Add a TextLine element to a parent (usually TextRegion).

        Args:
            parent: Parent element (TextRegion or Page)
            points: List of absolute coordinate points
            line_id: Optional line ID

        Returns:
            Created TextLine element
        """
        if line_id is None:
            line_id = f"line_{uuid.uuid4().hex[:8]}"

        text_line = ET.SubElement(parent, 'TextLine')
        text_line.set('id', line_id)

        # Add Coords
        coords = ET.SubElement(text_line, 'Coords')
        coords.set('points', self._points_to_string(points))

        return text_line

    def _prettify_xml(self, elem: ET.Element) -> str:
        """
        Return a pretty-printed XML string for the Element.

        Args:
            elem: XML element to prettify

        Returns:
            Pretty-printed XML string
        """
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def convert_page(self, yolo_txt: str, image_path: str, output_path: str = None) -> str:
        """
        Convert a single YOLO label file to PAGE-XML format with support for nested TextRegions.

        Args:
            yolo_txt: Path to the YOLO label txt file
            image_path: Path to the corresponding image file
            output_path: Optional output path for the XML file

        Returns:
            Path to the created XML file or XML string if no output_path
        """
        label_path = Path(yolo_txt)
        img_file = Path(image_path)

        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        if not img_file.exists():
            raise FileNotFoundError(f"Image file not found: {img_file}")

        # Get image dimensions
        with Image.open(img_file) as img:
            img_width, img_height = img.size

        # Create PAGE-XML structure
        root, page = self._create_page_element(img_file, img_width, img_height)

        # Read and process YOLO labels
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Categorize elements
        parent_text_regions = []  # TextRegions that can contain other TextRegions
        child_text_regions = []  # TextRegions that should be nested
        text_lines = []  # TextLines
        other_regions = []  # TableRegion, ImageRegion, etc.
        all_text_regions = []  # All TextRegions for TextLine assignment

        # First pass: Parse and categorize all elements
        for idx, line in enumerate(lines):
            if not line.strip():
                continue

            class_id, norm_points = self._parse_yolo_label(line)
            if not norm_points:
                continue

            # Denormalize points
            abs_points = self._denormalize_points(norm_points, img_width, img_height)

            # Get mapping for this class ID
            class_id_str = str(class_id)
            element_config = self.mapping.get(class_id_str, {})

            if not element_config:
                self.logger.warning(f"No mapping found for class ID {class_id}, skipping")
                continue

            element_type = element_config.get('element', 'TextLine')

            # Store element info based on type and parent settings
            if element_type == 'TextLine':
                text_lines.append({
                    'points': abs_points,
                    'id': f"l{idx:04d}",
                    'config': element_config
                })
            elif element_type == 'TextRegion':
                region_info = {
                    'points': abs_points,
                    'id': f"r{idx:04d}",
                    'type': element_config.get('type', None),
                    'config': element_config,
                    'element': None  # Will be set when created
                }

                # Check if this is a parent or child TextRegion
                parent_setting = element_config.get('parent', '')
                if parent_setting == 'TextRegion':
                    # This TextRegion should be nested in another TextRegion
                    child_text_regions.append(region_info)
                else:
                    # This is a top-level TextRegion (can contain others)
                    parent_text_regions.append(region_info)

            elif element_type in ['TableRegion', 'ImageRegion', 'GraphicRegion',
                                  'SeparatorRegion', 'NoiseRegion']:
                # Other regions - add directly to page
                self._add_region(page, element_type, abs_points,
                                 region_id=f"r{idx:04d}")

        # Second pass: Create parent TextRegions (top-level)
        self.logger.info(f"Processing {len(parent_text_regions)} parent TextRegions")
        for region_info in parent_text_regions:
            region_element = self._add_region(
                page,
                'TextRegion',
                region_info['points'],
                region_id=region_info['id'],
                region_type=region_info['type']
            )
            region_info['element'] = region_element
            all_text_regions.append(region_info)

        # Third pass: Assign child TextRegions to parent TextRegions
        orphan_child_regions = []
        self.logger.info(f"Processing {len(child_text_regions)} child TextRegions")

        for child_info in child_text_regions:
            # Find best parent TextRegion based on containment
            best_parent = self._find_best_parent(child_info['points'], parent_text_regions)

            if best_parent is not None:
                # Add as nested TextRegion
                child_element = self._add_region(
                    best_parent,
                    'TextRegion',
                    child_info['points'],
                    region_id=child_info['id'],
                    region_type=child_info['type']
                )
                child_info['element'] = child_element
                all_text_regions.append(child_info)
                self.logger.debug(f"Nested TextRegion {child_info['id']} in parent")
            else:
                # No suitable parent found - add to page level
                child_element = self._add_region(
                    page,
                    'TextRegion',
                    child_info['points'],
                    region_id=child_info['id'],
                    region_type=child_info['type']
                )
                child_info['element'] = child_element
                all_text_regions.append(child_info)
                orphan_child_regions.append(child_info['id'])
                self.logger.debug(f"Added TextRegion {child_info['id']} to page level (no parent found)")

        if orphan_child_regions:
            self.logger.info(f"{len(orphan_child_regions)} child TextRegions added to page level")

        # Fourth pass: Assign TextLines to TextRegions (any level)
        orphan_line_counter = 0
        self.logger.info(f"Processing {len(text_lines)} TextLines")

        for line_info in text_lines:
            line_points = line_info['points']
            line_id = line_info['id']

            # Find the most specific (smallest) TextRegion with significant overlap
            best_region = None
            best_region_area = float('inf')
            min_overlap_threshold = 0.1  # Minimum 10% overlap to consider

            # Calculate line area for overlap percentage
            try:
                line_poly = Polygon(line_points)
                line_area = line_poly.area if line_poly.is_valid else 0
            except:
                line_area = 0

            for region_info in all_text_regions:
                if region_info['element'] is None:
                    continue

                overlap = self._calculate_polygon_overlap(line_points, region_info['points'])

                # Check if overlap is significant (at least 10% of line area)
                if line_area > 0 and overlap > (line_area * min_overlap_threshold):
                    # Calculate region area
                    try:
                        region_poly = Polygon(region_info['points'])
                        region_area = region_poly.area if region_poly.is_valid else float('inf')
                    except:
                        region_area = float('inf')

                    # Choose the smallest region with significant overlap
                    if region_area < best_region_area:
                        best_region_area = region_area
                        best_region = region_info['element']
                        self.logger.debug(
                            f"Line {line_id}: Found smaller region with {overlap / line_area:.1%} overlap")

            if best_region is not None:
                # Add TextLine to the most specific (smallest) matching TextRegion
                self._add_text_line(best_region, line_points, line_id=line_id)
            else:
                # No overlap found - create a new TextRegion for this TextLine
                parent_type = line_info['config'].get('parent_type', 'paragraph')
                new_region = self._add_region(
                    page,
                    'TextRegion',
                    line_points,
                    region_id=f"r_orphan{orphan_line_counter:04d}",
                    region_type=parent_type
                )
                self._add_text_line(new_region, line_points, line_id=line_id)
                orphan_line_counter += 1

        # Log statistics
        if text_lines:
            self.logger.info(f"TextLine assignment: {len(text_lines) - orphan_line_counter} to existing regions, "
                             f"{orphan_line_counter} in new regions")

        # Generate XML string
        xml_string = self._prettify_xml(root)

        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_string)
            self.logger.info(f"Created PAGE-XML: {output_path}")
            return str(output_path)
        else:
            return xml_string

    def convert_folder(self, labels_folder: str, images_folder: str, output_folder: str = None):
        """
        Convert all YOLO txt files in a folder to PAGE-XML format.

        Args:
            labels_folder: Folder containing YOLO txt label files
            images_folder: Folder containing corresponding image files
            output_folder: Output folder for PAGE-XML files (default: labels_folder/page_xml)
        """
        labels_folder = Path(labels_folder)
        images_folder = Path(images_folder)

        if not labels_folder.exists():
            raise FileNotFoundError(f"Labels folder not found: {labels_folder}")

        if not images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {images_folder}")

        # Set output folder
        if output_folder is None:
            output_folder = labels_folder / 'page_xml'
        else:
            output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Get all txt files
        txt_files = list(labels_folder.glob('*.txt'))

        if not txt_files:
            self.logger.warning(f"No txt files found in {labels_folder}")
            return

        self.logger.info(f"Found {len(txt_files)} txt files to convert")

        # Process each txt file
        converted = 0
        failed = 0

        for txt_file in txt_files:
            try:
                # Find corresponding image with same base name
                img_file = None
                for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp']:
                    potential_img = images_folder / f"{txt_file.stem}{ext}"
                    if potential_img.exists():
                        img_file = potential_img
                        break

                if not img_file:
                    self.logger.warning(f"No image found for label: {txt_file.name}")
                    failed += 1
                    continue

                # Convert to PAGE-XML
                output_path = output_folder / f"{txt_file.stem}.xml"
                self.convert_page(str(txt_file), str(img_file), str(output_path))
                converted += 1

            except Exception as e:
                self.logger.error(f"Failed to convert {txt_file.name}: {str(e)}")
                failed += 1

        self.logger.info(f"Conversion complete: {converted} successful, {failed} failed")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert YOLO segmentation format to PAGE-XML')
    parser.add_argument('labels', type=str, help='YOLO labels folder or single txt file')
    parser.add_argument('images', type=str, help='Images folder or single image file')
    parser.add_argument('--output', type=str, help='Output folder for PAGE-XML files or single XML file')
    parser.add_argument('--mapping', type=str, help='Path to JSON mapping configuration file')

    args = parser.parse_args()

    # Create converter
    converter = YoloPageConverter(mapping_file=args.mapping)

    # Check if single file or folder conversion
    labels_path = Path(args.labels)
    images_path = Path(args.images)

    if labels_path.is_file() and labels_path.suffix == '.txt':
        # Single file conversion
        if not images_path.is_file():
            parser.error("For single file conversion, provide a single image file")
        output_path = args.output if args.output else labels_path.with_suffix('.xml')
        converter.convert_page(str(labels_path), str(images_path), output_path)

    elif labels_path.is_dir() and images_path.is_dir():
        # Folder conversion
        converter.convert_folder(str(labels_path), str(images_path), args.output)

    else:
        parser.error("Provide either two files (txt and image) or two folders (labels and images)")