import os
import cv2
import json
import requests
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import logging
import uuid
import tempfile
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for interacting with OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, messages: List[Dict], max_retries: int = 3) -> Tuple[Optional[Dict], str]:
        """Generate response using OpenAI API with retry logic"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 512,
            "top_p": 0.95,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    generated_text = response_data['choices'][0]['message']['content']
                    
                    if not generated_text or not generated_text.strip():
                        logger.warning(f"Empty response received on attempt {attempt + 1}, retrying...")
                        continue
                    
                    json_result = self._extract_json_from_response(generated_text)
                    
                    if json_result:
                        validated_json = self._validate_and_fix_json(json_result)
                        logger.info("Successfully extracted and validated JSON from API response")
                        return validated_json, generated_text
                    else:
                        logger.warning(f"Failed to extract JSON from response on attempt {attempt + 1}")
                
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:
                        break
                    time.sleep(1)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                break
        
        return None, ""
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from API response"""
        import re
        
        if not response_text or not response_text.strip():
            return None
            
        response_text = response_text.strip()
        
        # Try parsing entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Handle code blocks
        if '```' in response_text:
            code_block_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
            ]
            
            for pattern in code_block_patterns:
                matches = re.findall(pattern, response_text, re.DOTALL | re.MULTILINE)
                for match in matches:
                    try:
                        return json.loads(match.strip())
                    except json.JSONDecodeError:
                        continue
        
        # Try regex patterns
        json_patterns = [
            r'(\{(?:[^{}]|{[^{}]*})*\})',
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'JSON:\s*(\{.*?\})',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                
                cleaned_match = match.strip()
                if cleaned_match.startswith('{') and cleaned_match.endswith('}'):
                    try:
                        return json.loads(cleaned_match)
                    except json.JSONDecodeError:
                        continue
        
        return None
    
    def _validate_and_fix_json(self, json_data: Dict) -> Dict:
        """Ensure JSON contains all required fields"""
        required_fields = [
            "specimen_family",
            "specimen_genus", 
            "country_country",
            "locality_locality",
            "collect_date_collect_date",
            "collector_collector",
            "rest_of_text"
        ]
        
        clean_json = {}
        for field in required_fields:
            clean_json[field] = json_data.get(field, "")
            if not isinstance(clean_json[field], str):
                clean_json[field] = str(clean_json[field]) if clean_json[field] is not None else ""
        
        return clean_json

class HerbariumProcessor:
    """Main processor for herbarium specimen images"""
    
    def __init__(self, yolo_model_path: str, trocr_model_path: str, openai_api_key: str, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        weights_path = hf_hub_download(repo_id=yolo_model_path,
        filename="sheet-component.pt")
        self.yolo_model = YOLO(weights_path)
        self.trocr_processor = None
        self.trocr_model = None
        self.openai_client = OpenAIClient(openai_api_key)
        self.craft_model = None
        
        # Target classes for text extraction
        self.target_classes = ['handwritten data', 'institutional label', 'annotation label']
        
        # Load models
        self._load_craft_model()
        self._load_trocr_model(trocr_model_path)
    
    def _load_craft_model(self):
        """Load CRAFT model for text detection"""
        try:
            from hezar.models import Model
            self.craft_model = Model.load("hezarai/CRAFT", device=self.device)
            logger.info("CRAFT model loaded successfully")
        except ImportError:
            logger.error("Hezar package not found. Please install with 'pip install hezar'")
            raise
    
    def _load_trocr_model(self, model_path: str):
        """Load TrOCR model for text recognition"""
        try:
            logger.info(f"Loading TrOCR model from {model_path}...")
            self.trocr_processor = TrOCRProcessor.from_pretrained(model_path)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path)
            self.trocr_model.to(self.device)
            self.trocr_model.eval()
            logger.info("TrOCR model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading TrOCR model: {e}")
            raise
    def detect_components(self, img_path: str, confidence_threshold: float = 0.7) -> Tuple[np.ndarray, List[Dict]]:
        """Detect herbarium sheet components using YOLO with confidence filtering"""
        img = cv2.imread(img_path)
        results = self.yolo_model(img)
        
        filtered_results = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls.item())
                class_name = self.yolo_model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf.item()
                
                # Filter by confidence threshold
                if confidence >= confidence_threshold:
                    filtered_results.append({
                        'class': class_name,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
        
        logger.info(f"Detected {len(filtered_results)} components with confidence >= {confidence_threshold}")
        return img, filtered_results
    
    def detect_text_regions(self, component_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions using CRAFT model"""
        if isinstance(component_img, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = component_img
        
        outputs = self.craft_model.predict(pil_img)
        bboxes = outputs[0]["boxes"]
        
        correct_bboxes = []
        for bbox in bboxes:
            if bbox is None:
                continue
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            if x1 == x2:
                x2 = x1 + 1
            if y1 == y2:
                y2 = y1 + 1
            
            correct_bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return correct_bboxes
    
    def order_words_optimal(self, word_images: List[Dict]) -> List[Dict]:
        """Order words using DBSCAN clustering for line detection"""
        if not word_images:
            return word_images
        
        # Extract coordinates and heights
        coordinates = []
        heights = []
        
        for word in word_images:
            bbox = word['bbox']
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            height = y2 - y1
            
            coordinates.append([center_x, center_y])
            heights.append(height)
            
            word['center_x'] = center_x
            word['center_y'] = center_y
            word['height'] = height
        
        coordinates = np.array(coordinates)
        median_height = np.median(heights)
        
        # Line detection using DBSCAN
        line_clusters = self._detect_lines_with_dbscan(coordinates, median_height)
        
        # Group words by detected lines
        lines = self._group_words_by_lines(word_images, line_clusters)
        
        # Sort lines and words
        ordered_words = self._sort_lines_and_words(lines)
        
        return ordered_words
    
    def _detect_lines_with_dbscan(self, coordinates: np.ndarray, median_height: float) -> np.ndarray:
        """DBSCAN clustering for line detection"""
        if len(coordinates) < 2:
            return np.array([0] * len(coordinates))
        
        y_coords = coordinates[:, 1].reshape(-1, 1)
        eps = median_height * 0.6
        
        clustering = DBSCAN(eps=eps, min_samples=1, metric='euclidean').fit(y_coords)
        labels = clustering.labels_
        
        # Handle noise points
        noise_indices = np.where(labels == -1)[0]
        for noise_idx in noise_indices:
            noise_y = y_coords[noise_idx][0]
            min_distance = float('inf')
            closest_label = 0
            
            for i, label in enumerate(labels):
                if label != -1:
                    distance = abs(y_coords[i][0] - noise_y)
                    if distance < min_distance:
                        min_distance = distance
                        closest_label = label
            
            if min_distance <= eps * 1.5:
                labels[noise_idx] = closest_label
            else:
                labels[noise_idx] = max(labels) + 1 if len(labels) > 0 else 0
        
        return labels
    
    def _group_words_by_lines(self, word_images: List[Dict], line_labels: np.ndarray) -> List[List[Dict]]:
        """Group words into lines based on clustering results"""
        lines = {}
        
        for word, label in zip(word_images, line_labels):
            if label not in lines:
                lines[label] = []
            lines[label].append(word)
        
        return list(lines.values())
    
    def _sort_lines_and_words(self, lines: List[List[Dict]]) -> List[Dict]:
        """Sort lines top-to-bottom and words within lines left-to-right"""
        ordered_words = []
        order_index = 0
        
        # Sort lines by Y position
        lines.sort(key=lambda line: np.mean([w['center_y'] for w in line]))
        
        for line in lines:
            # Sort words in line by X coordinate
            line.sort(key=lambda word: word['center_x'])
            
            for word in line:
                word['order'] = order_index
                ordered_words.append(word)
                order_index += 1
        
        return ordered_words
    
    def crop_word_images(self, component_img: np.ndarray, text_regions: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Crop word images from component based on text regions"""
        word_images = []
        
        for i, bbox in enumerate(text_regions):
            x1, y1, x2, y2 = bbox
            padding = 2
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(component_img.shape[1], x2 + padding)
            y2 = min(component_img.shape[0], y2 + padding)
            
            word_img = component_img[y1:y2, x1:x2]
            
            if word_img.size > 0:
                word_images.append({
                    'image': word_img,
                    'bbox': (x1, y1, x2, y2),
                    'detection_index': i,
                    'order': None
                })
        
        # Apply optimal ordering
        word_images = self.order_words_optimal(word_images)
        return word_images
    
    def recognize_text_trocr(self, word_img: np.ndarray) -> str:
        """Recognize text using TrOCR model"""
        try:
            if self.trocr_processor is None or self.trocr_model is None:
                return "[MODEL_NOT_LOADED]"
            
            # Convert to PIL Image
            if isinstance(word_img, np.ndarray):
                if len(word_img.shape) == 3 and word_img.shape[2] == 3:
                    pil_img = Image.fromarray(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
                elif len(word_img.shape) == 2:
                    word_img_rgb = cv2.cvtColor(word_img, cv2.COLOR_GRAY2RGB)
                    pil_img = Image.fromarray(word_img_rgb)
                else:
                    return "[INVALID_IMAGE]"
            else:
                pil_img = word_img
            
            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                return "[EMPTY_IMAGE]"
            
            # Process with TrOCR
            pixel_values = self.trocr_processor(images=pil_img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values)
            
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error in TrOCR recognition: {e}")
            return "[TROCR_ERROR]"
    
    def extract_text_from_components(self, img: np.ndarray, components: List[Dict]) -> Dict:
        """Extract text from target components"""
        image_name = "specimen"
        json_data = {
            "id": image_name,
            "components": []
        }
        
        for i, component in enumerate(components):
            component_class = component['class']
            bbox = component['bbox']
            x1, y1, x2, y2 = bbox
            confidence = component['confidence']
            
            # Crop component
            component_img = img[y1:y2, x1:x2]
            component_id = str(uuid.uuid4())
            temp_dir = tempfile.mkdtemp(prefix="herbarium_components_")
            image_filename = f"component_{component_id}.jpg"
            image_path = os.path.join(temp_dir, image_filename)
            cv2.imwrite(image_path, component_img)
            component_data = {
                "class": component_class,
                "bbox": list(bbox),
                "confidence": float(confidence),
                "crop_id": f"{image_name}_{component_class}_{i}",
                "image_id": component_id,
            }
            
            # Only process text for target classes
            if component_class in self.target_classes:
                component_data["words"] = []
                
                try:
                    # Detect text regions
                    text_regions = self.detect_text_regions(component_img)
                    
                    # Crop word images and order them
                    word_images = self.crop_word_images(component_img, text_regions)
                    
                    for word_data in word_images:
                        word_img = word_data['image']
                        word_bbox = word_data['bbox']
                        word_order = word_data['order']
                        
                        word_id = f"{component_data['crop_id']}_word_{word_order:03d}"
                        
                        # Recognize text with TrOCR
                        text = self.recognize_text_trocr(word_img)
                        
                        component_data["words"].append({
                            "id": word_id,
                            "bbox": list(word_bbox),
                            "text": text,
                            "order": word_order,
                            "detection_index": word_data.get('detection_index', -1)
                        })
                    
                    # Create full text
                    invalid_texts = ["[TROCR_ERROR]", "[EMPTY_IMAGE]", "[INVALID_IMAGE]", "[MODEL_NOT_LOADED]"]
                    valid_texts = [w['text'] for w in 
                                 sorted(component_data["words"], key=lambda x: x['order'])
                                 if w['text'] not in invalid_texts]
                    component_data["full_text"] = ' '.join(valid_texts)
                    
                except Exception as e:
                    logger.error(f"Error processing component {component_data['crop_id']}: {e}")
                    component_data["error"] = str(e)
                    component_data["words"] = []
                    component_data["full_text"] = "[PROCESSING_ERROR]"
            
            json_data["components"].append(component_data)
        
        return json_data
    
    def prepare_llm_prompt(self, specimen_data: Dict[str, str]) -> List[Dict[str, str]]:
        """Prepare prompt for LLM processing"""
        system_prompt = """Tu es un expert botaniste spécialisé dans la lecture et la correction des données d'étiquettes d'herbier, souvent issues de l'OCR. Tu dois extraire et corriger les informations structurées, même à partir de textes partiellement lisibles ou ambigus.

RÈGLES CRITIQUES À RESPECTER ABSOLUMENT:
1. Tu dois retourner UNIQUEMENT un objet JSON valide, sans aucun texte explicatif avant ou après
2. Pas de texte d'introduction, pas d'explication, pas de commentaires – SEULEMENT le JSON
3. Le JSON doit commencer par { et finir par }
4. Ne pas utiliser de balises markdown ou de formatage
5. Respecter exactement la structure fournie avec EXACTEMENT ces 7 champs

CHAMPS OBLIGATOIRES (tous requis, même si vides):
• specimen_family: Nom de la famille botanique (ex: "Iridaceae")
• specimen_genus: Nom du genre (ex: "Iris") 
• country_country: Nom complet du pays (ex: "France")
• locality_locality: Localité précise (ville, région, etc.)
• collect_date_collect_date: Date au format "JJ/MM/AAAA" ou "MM/AAAA" si le jour est inconnu
• collector_collector: Nom du collecteur au format "Nom, Prénom/Initiale"
• rest_of_text: Tout autre texte non classé dans les champs précédents

RÈGLES D'INTERPRÉTATION ET DE CORRECTION :
- Corriger les erreurs typiques de l'OCR (lettres inversées, accents, coupures, etc.)
- Ne JAMAIS inventer d'information non présente, mais corriger si c'est manifestement une faute de lecture OCR
- Convertir toutes les dates au bon format
- Pour les noms botaniques : vérifier la validité et la syntaxe
- Ne pas dupliquer l'information dans plusieurs champs
- rest_of_text doit inclure tout contenu pertinent non utilisé"""

        json_template = {
            "specimen_family": "",
            "specimen_genus": "",
            "country_country": "",
            "locality_locality": "",
            "collect_date_collect_date": "",
            "collector_collector": "",
            "rest_of_text": ""
        }

        user_prompt = f"""Analyse et corrige les textes suivants selon les règles données. Ne retourne que le JSON strictement conforme.

STRUCTURE EXACTE À REMPLIR (7 champs) :
{json.dumps(json_template, indent=2, ensure_ascii=False)}

TEXTES À ANALYSER :
---
Texte d'annotation : {specimen_data.get('annotation_label_text', '')}
---
Texte manuscrit : {specimen_data.get('handwritten_data_text', '')}
---
Texte institutionnel : {specimen_data.get('institutional_label_text', '')}
---

RAPPEL FINAL : Le résultat doit être un seul JSON valide sans explication, débutant par {{ et finissant par }}."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def extract_structured_data(self, json_data: Dict) -> Dict:
        """Extract structured data using LLM"""
        # Extract text components
        specimen_data = {
            'annotation_label_text': '',
            'handwritten_data_text': '',
            'institutional_label_text': ''
        }
        
        for component in json_data.get('components', []):
            component_class = component.get('class', '').lower()
            full_text = component.get('full_text', '')
            
            if 'annotation label' in component_class:
                specimen_data['annotation_label_text'] += (' ' + full_text) if specimen_data['annotation_label_text'] else full_text
            elif 'handwritten data' in component_class:
                specimen_data['handwritten_data_text'] += (' ' + full_text) if specimen_data['handwritten_data_text'] else full_text
            elif 'institutional label' in component_class:
                specimen_data['institutional_label_text'] += (' ' + full_text) if specimen_data['institutional_label_text'] else full_text
        
        # Prepare prompt and get LLM response
        messages = self.prepare_llm_prompt(specimen_data)
        structured_data, raw_response = self.openai_client.generate_response(messages)
        
        return structured_data or {}
    
    def process_single_image(self, img_path: str) -> Dict:
        """Process a single image"""
        try:
            logger.info(f"Processing image: {img_path}")
            
            # Step 1: Component detection
            img, components = self.detect_components(img_path)
            
            # Step 2-4: Text extraction
            json_data = self.extract_text_from_components(img, components)
            
            # Step 5: LLM processing
            structured_data = self.extract_structured_data(json_data)
            
            result = {
                "status": "success",
                "image_path": img_path,
                "json_data": json_data,
                "structured_data": structured_data
            }
            
            logger.info(f"Successfully processed image: {img_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return {
                "status": "error",
                "image_path": img_path,
                "error": str(e)
            }

class HerbariumPipeline:
    """Main pipeline class that orchestrates the entire process"""
    
    def __init__(self, config: Dict):
        """Initialize pipeline with configuration"""
        self.config = config
        self.processor = HerbariumProcessor(
            yolo_model_path=config['yolo_model_path'],
            trocr_model_path=config['trocr_model_path'],
            openai_api_key=config['openai_api_key'],
            device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        logger.info("Herbarium pipeline initialized successfully")
    
    def process_single_image(self, img_path: str) -> Dict:
        """Process a single herbarium specimen image"""
        return self.processor.process_single_image(img_path)
    
    def validate_config(self) -> Dict[str, bool]:
        validation_results = {}

        # Check YOLO model
        try:
            from huggingface_hub import repo_exists
            validation_results['yolo_model'] = repo_exists(self.config['yolo_model_path'])
        except:
            validation_results['yolo_model'] = False

        # Check TrOCR model
        try:
            from huggingface_hub import repo_exists
            validation_results['trocr_model'] = repo_exists(self.config['trocr_model_path'])
        except:
            validation_results['trocr_model'] = False
        
        # Check OpenAI API key
        try:
            headers = {
                "Authorization": f"Bearer {self.config['openai_api_key']}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=10
            )
            validation_results['openai_api'] = response.status_code == 200
        except:
            validation_results['openai_api'] = False
        
        # Check CRAFT dependencies
        try:
            from hezar.models import Model
            validation_results['craft_dependencies'] = True
        except ImportError:
            validation_results['craft_dependencies'] = False
        
        return validation_results