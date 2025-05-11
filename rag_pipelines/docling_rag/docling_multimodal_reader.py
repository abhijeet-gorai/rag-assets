import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from PIL import Image
from utils.chat_image import ChatWithImage
from prompts import rag_prompts
from langchain_core.documents import Document
from .config_schema import VisionLLMConfig
from utils.rate_limiter import RateLimiter

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, GroupItem, SectionHeaderItem
from docling_core.types.doc.labels import DocItemLabel
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

logger = logging.getLogger(__name__)


def base64_to_pil(base64_string: str) -> Image.Image:
    """
    Convert a base64 encoded string to a PIL Image.

    :param base64_string: Base64 encoded image string.
    :return: PIL Image object.
    """
    image_data = base64.b64decode(base64_string)
    image_bytes = BytesIO(image_data)
    pil_image = Image.open(image_bytes)
    return pil_image


class DoclingMultimodalReader:
    """
    A multimodal PDF reader pipeline that processes PDF elements and provides options for grouping:
    
      - "group_by_page": All elements on a page are grouped together. When images are not kept separate, 
          non-image text is combined into a single element and all image/table elements are collected under an "images" key.
          When images are kept separate, the page is segmented at every image/table boundary.
      
      - "group_by_title": All elements between two Title elements are grouped together. In this case the title is saved
          in metadata under the "title" key. The grouping logic for images is the same as in the page-based grouping.
      
      - "raw": No grouping is applied; the processed elements are returned as a flat list.
    """
    ELEMENT_TYPE_IMAGE = "Image"
    ELEMENT_TYPE_TABLE = "Table"
    ELEMENT_TYPE_TITLE = "Title"

    def __init__(
        self,
        vision_llm_config: VisionLLMConfig = {},
        context_window: int = 4,
        image_processing_system_prompt: Optional[str] = None,
        image_processing_user_prompt: Optional[str] = None,
        table_preprocessing_system_prompt: Optional[str] = None,
        table_preprocessing_user_prompt: Optional[str] = None,
        max_workers: int = 32,
        max_requests_per_second: int = 8
    ) -> None:
        """
        Initialize the PDF reader pipeline.

        :param vision_llm_config: Config for Vision LLM.
        :param context_window: Number of text elements before and after an image/table element to use as context.
        :param image_processing_system_prompt: System prompt for processing images.
        :param image_processing_user_prompt: User prompt (template) for processing images; should include a placeholder
               (e.g. {context}) to inject context.
        :param table_preprocessing_system_prompt: System prompt for processing tables.
        :param table_preprocessing_user_prompt: User prompt (template) for processing tables; should include a placeholder
               (e.g. {context}) to inject context.
        :param max_workers: Maximum number of threads to use for making requests to LLM. Defaults to 32
        :param max_requests_per_second: Maximum number of requests per second supported by LLM. Defaults to 8
        """
        self._image_llm = ChatWithImage(**vision_llm_config)
        self.image_processing_system_prompt = (
            image_processing_system_prompt or rag_prompts.image_processing_system_prompt
        )
        self.image_processing_user_prompt = (
            image_processing_user_prompt or rag_prompts.image_processing_user_prompt
        )
        self.table_preprocessing_system_prompt = (
            table_preprocessing_system_prompt or rag_prompts.table_preprocessing_system_prompt
        )
        self.table_preprocessing_user_prompt = (
            table_preprocessing_user_prompt or rag_prompts.table_preprocessing_user_prompt
        )
        self.context_window = context_window
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(max_calls=max_requests_per_second, period=1)

    def check_image_quality(self, base64_image: str) -> bool:
        """
        Check whether an image meets minimum quality and size requirements.

        :param base64_image: Base64 encoded image.
        :return: True if image quality is acceptable; False otherwise.
        """
        try:
            image = base64_to_pil(base64_image)
            width, height = image.size
            min_size = (30, 30)
            min_group_size = (150, 150)
            if width <= min_size[0] or height <= min_size[1] or (width <= min_group_size[0] and height <= min_group_size[1]):
                return False
        except Exception as e:
            logger.error("Failed to check image quality: %s", e, exc_info=True)
        return True

    def read_document(
        self,
        file_path: str,
        group_mode: str = "group_by_title",
        keep_images_tables_separate: bool = False,
    ) -> List[Document] | List[Dict[str, Any]]:
        """
        Process a PDF file and return its content with multimodal processing.

        The PDF is partitioned into elements, context is attached to image/table elements, and those elements are
        processed concurrently via a multimodal LLM. Finally, the results are grouped according to the grouping mode.

        :param file_path: Path to the PDF file.
        :param group_mode: Grouping mode; either "group_by_title", or "raw".
        :param keep_images_tables_separate: If True, image/table elements are output as separate elements (splitting the group)
               rather than merged into the grouped output.
        :return: A list of Documents. Each Document has:
                 - "page_content": The combined text (or summary) for the segment.
                 - "metadata": For page groups, keys include "filename", "page_number", and "images".
                   For title groups, a "title" key is added.
                   For raw mode, keys include "filename", "page_number", "image_base64", and "image_text".
        :raises ValueError: If group_mode is invalid.
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.images_scale = 2
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_page_images = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.MPS
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(file_path)

        processed_elements: List[Dict[str, Any]] = []
        for element, _level in result.document.iterate_items():
            elem = {"text": "", "context": "", "metadata": {"image_base64": "", "filename": file_path}}
            if isinstance(element, TableItem):
                elem["type"] = self.ELEMENT_TYPE_TABLE
                table_content = element.export_to_markdown()
                elem["text"] = table_content
                processed_elements.append(elem)
            elif isinstance(element, PictureItem):
                elem["type"] = self.ELEMENT_TYPE_IMAGE
                image_base64 = str(element.image.uri)
                elem["metadata"]["image_base64"] = image_base64
                processed_elements.append(elem)
            else:
                if isinstance(element, SectionHeaderItem) or element.label in [DocItemLabel.TITLE]:
                    elem["type"] = self.ELEMENT_TYPE_TITLE
                else:
                    elem["type"] = "Text"
                elem["text"] = element.text
                if elem["text"]:
                    processed_elements.append(elem)

        # Attach context to image/table elements.
        for idx, elem in enumerate(processed_elements):
            if elem.get("type") in (self.ELEMENT_TYPE_IMAGE, self.ELEMENT_TYPE_TABLE):
                context = self.get_image_or_table_context(idx, processed_elements)
                elem["context"] = context

        # Concurrently process all image/table elements.
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for elem in processed_elements:
                if elem.get("type") in [self.ELEMENT_TYPE_IMAGE]:
                    futures.append(executor.submit(self.process_image_element, elem))
                elif elem.get("type") in [self.ELEMENT_TYPE_TABLE]:
                    futures.append(executor.submit(self.process_table_element, elem))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
                future.result()

        # Group the elements according to the selected group_mode.
        if group_mode == "group_by_title":
            output = self.group_by_title(processed_elements, keep_images_tables_separate)
        elif group_mode == "raw":
            output = processed_elements
        else:
            raise ValueError("Invalid group_mode, must be 'group_by_title', or 'raw'.")
        return output

    def get_image_or_table_context(self, ele_id: int, elements: List[Dict[str, Any]]) -> str:
        """
        Extract context for an image/table element from surrounding text elements.

        :param ele_id: Index of the image/table element.
        :param elements: List of all elements.
        :return: Combined context string from preceding and following text elements.
        """
        context_parts = []
        # Preceding text elements.
        start = max(0, ele_id - self.context_window)
        for j in range(start, ele_id):
            if elements[j].get("type") not in (self.ELEMENT_TYPE_IMAGE, self.ELEMENT_TYPE_TABLE):
                text = elements[j].get("text")
                if text:
                    context_parts.append(text)
        # Following text elements.
        end = min(len(elements), ele_id + 1 + self.context_window)
        for j in range(ele_id + 1, end):
            if elements[j].get("type") not in (self.ELEMENT_TYPE_IMAGE, self.ELEMENT_TYPE_TABLE):
                text = elements[j].get("text")
                if text:
                    context_parts.append(text)
        return "\n".join(context_parts)

    def process_image_element(self, elem: Dict[str, Any]) -> None:
        """
        Process a single image or table element using multimodal LLM prompts.

        Depending on the element type, the appropriate system and user prompts are selected (with context injected)
        and the image is sent to the multimodal LLM. The returned summary is stored in the element.

        :param elem: The element dictionary to process.
        """
        self.rate_limiter.wait()
        metadata = elem.get("metadata", {})
        image_base64 = metadata.get("image_base64", "")
        if not image_base64:
            return
        context = elem.get("context", "")
        system_prompt = self.image_processing_system_prompt
        user_prompt = self.image_processing_user_prompt.format(context=context)

        try:
            summary = self._image_llm.chat_with_image(
                images=[image_base64],
                prompt=user_prompt,
                system_message=system_prompt,
                convert_images_to_base64=False,
            )
        except Exception as e:
            logger.error("Failed to process image element: %s", e)
            raise

        # Save the summary
        elem["summary"] = summary

    def process_table_element(self, elem: Dict[str, Any]) -> None:
        """
        Process a single image or table element using multimodal LLM prompts.

        Depending on the element type, the appropriate system and user prompts are selected (with context injected)
        and the image is sent to the multimodal LLM. The returned summary is stored in the element.

        :param elem: The element dictionary to process.
        """
        self.rate_limiter.wait()
        table_data = elem.get("text", "")
        if not table_data:
            return
        context = elem.get("context", "")
        system_prompt = self.table_preprocessing_system_prompt
        user_prompt = self.table_preprocessing_user_prompt.format(context=context)
        user_prompt = f"{user_prompt}\nTable Data:\n{table_data}"
        try:
            summary = self._image_llm.chat_with_image(
                prompt=user_prompt,
                system_message=system_prompt
            )
        except Exception as e:
            logger.error("Failed to process image element: %s", e)
            raise

        # Save the summary
        elem["summary"] = summary

    def group_by_title(
        self, elements: List[Dict[str, Any]], keep_images_tables_separate: bool
    ) -> List[Document]:
        """
        Group processed elements by Title.

        In group_by_title mode, all elements between two Title elements are combined into a group.
        The group's title is saved in metadata under the "title" key.
        If keep_images_tables_separate is False, non-image elements are merged and image/table elements are gathered
        into an "images" list. Otherwise, the group is segmented so that text segments and image/table elements
        are output separately.

        :param elements: List of processed elements.
        :param keep_images_tables_separate: Whether to output image/table elements as separate outputs.
        :return: A list of grouped output dictionaries.
        """
        groups = []
        current_group = {"title": "", "elements": []}
        for elem in elements:
            if elem.get("type") == self.ELEMENT_TYPE_TITLE:
                if current_group["elements"]:
                    groups.append(current_group)
                current_group = {"title": elem.get("text", ""), "elements": [elem]}
            else:
                current_group["elements"].append(elem)
        if current_group["elements"]:
            groups.append(current_group)

        outputs = []
        for group in groups:
            title = group["title"]
            elems = group["elements"]
            if not keep_images_tables_separate:
                text_parts = []
                images = []
                tables = []
                for elem in elems:
                    if elem.get("type") in [self.ELEMENT_TYPE_IMAGE]:
                        images.append({
                            "image_base64": elem.get("metadata", {}).get("image_base64", ""),
                            "image_text": elem.get("text", ""),
                        })
                        text_val = elem.get("summary", "")
                    elif elem.get("type") in [self.ELEMENT_TYPE_TABLE]:
                        tables.append({
                            "table_text": elem.get("text", ""),
                        })
                        text_val = elem.get("summary", "")
                    else:
                        text_val = elem.get("text", "")
                    if text_val:
                        text_parts.append(text_val)
                combined_text = "\n".join(text_parts).strip()
                metadata_out = {"filename": elem.get("metadata", {}).get("filename", ""), "title": title, "images": images, "tables": tables}
                outputs.append({"page_content": combined_text, "metadata": metadata_out})
            else:
                current_text_parts = []
                for elem in elems:
                    if elem.get("type") in [self.ELEMENT_TYPE_IMAGE]:
                        if current_text_parts:
                            combined_text = "\n".join(current_text_parts).strip()
                            metadata_out = {"filename": elem.get("metadata", {}).get("filename", ""), "title": title, "images": [], "tables": []}
                            outputs.append({"page_content": combined_text, "metadata": metadata_out})
                            current_text_parts = []
                        image_data = {
                            "image_base64": elem.get("metadata", {}).get("image_base64", ""),
                            "image_text": elem.get("text", ""),
                        }
                        metadata_out = {"filename": elem.get("metadata", {}).get("filename", ""), "title": title, "images": [image_data], "tables": []}
                        outputs.append({"page_content": elem.get("summary", ""), "metadata": metadata_out})
                    elif elem.get("type") in [self.ELEMENT_TYPE_TABLE]:
                        if current_text_parts:
                            combined_text = "\n".join(current_text_parts).strip()
                            metadata_out = {"filename": elem.get("metadata", {}).get("filename", ""), "title": title, "images": [], "tables": []}
                            outputs.append({"page_content": combined_text, "metadata": metadata_out})
                            current_text_parts = []
                        table_data = {
                            "table_text": elem.get("text", ""),
                        }
                        metadata_out = {"filename": elem.get("metadata", {}).get("filename", ""), "title": title, "images": [], "tables": [table_data]}
                        outputs.append({"page_content": elem.get("summary", ""), "metadata": metadata_out})
                    else:
                        text_val = elem.get("text", "")
                        if text_val:
                            current_text_parts.append(text_val)
                if current_text_parts:
                    combined_text = "\n".join(current_text_parts).strip()
                    metadata_out = {"filename": elem.get("metadata", {}).get("filename", ""), "title": title, "images": []}
                    outputs.append({"page_content": combined_text, "metadata": metadata_out})
        outputs = [Document(**output) for output in outputs]
        return outputs
