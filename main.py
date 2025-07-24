from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import subprocess
import base64
import io
import re
import tempfile
import shutil
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image as PILImage
import google.generativeai as genai
from datetime import datetime
import vertexai
from vertexai.preview.vision_models import Image, ImageGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---
class GenerateImageRequest(BaseModel):
    prompt: str
    num_images: int = 1
    aspect_ratio: str = "16:9"

class AnalyzeImageRequest(BaseModel):
    prompt: str = ""
    context: str = ""

class AgentResponse(BaseModel):
    status: str
    action: str
    message: str
    data: Optional[Dict] = None

# --- Configuration Class ---
class ImageAgentConfig:
    """Configuration class for the Image Agent"""
    
    def __init__(self):
        # Google Cloud Project Configuration
        self.GCP_PROJECT_ID = "vertex-ai-466616"  # Replace with your project ID
        self.GCP_LOCATION = "us-central1"
        
        # Image Generation Settings
        self.OUTPUT_DIR = "static/images"  # Directory to save generated images
        self.TEXT_OUTPUT_DIR = "static/analysis"  # Directory to save analysis results
        self.UPLOAD_DIR = "uploads"  # Directory for uploaded images
        self.DEFAULT_ASPECT_RATIO = "16:9"
        self.DEFAULT_SAFETY_LEVEL = "block_few"
        
        # Vertex AI Models
        self.IMAGEN_MODEL = "imagen-4.0-generate-preview-06-06"
        self.VISION_MODEL = "gemini-2.5-pro"
        
        # File settings
        self.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        self.ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

# --- Prompt Classification ---
class PromptClassifier:
    """Intelligent prompt classification to determine the required action"""
    
    def __init__(self):
        self.generation_keywords = {
            'create', 'generate', 'make', 'draw', 'design', 'paint', 'sketch', 
            'produce', 'build', 'craft', 'compose', 'render', 'illustrate',
            'visualize', 'imagine', 'picture', 'show me', 'create an image',
            'generate image', 'make picture', 'draw me', 'design a', 'paint a'
        }
        
        self.analysis_keywords = {
            'analyze', 'describe', 'explain', 'what is', 'what are', 'tell me about',
            'identify', 'recognize', 'detect', 'find', 'look at', 'examine',
            'inspect', 'review', 'understand', 'see', 'observe', 'read',
            'what do you see', 'describe this', 'analyze image', 'what is in',
            'caption', 'summarize', 'interpret', 'decode'
        }
    
    def classify_prompt(self, prompt: str, has_image_attachment: bool = False) -> Dict:
        """Classify the user's prompt to determine the required action"""
        prompt_lower = prompt.lower().strip()
        
        generation_score = 0
        analysis_score = 0
        
        # Check for explicit keywords
        for keyword in self.generation_keywords:
            if keyword in prompt_lower:
                generation_score += 2
                if keyword in prompt_lower[:20]:
                    generation_score += 1
        
        for keyword in self.analysis_keywords:
            if keyword in prompt_lower:
                analysis_score += 2
                if keyword in prompt_lower[:20]:
                    analysis_score += 1
        
        # Strong indicators based on context
        if has_image_attachment:
            analysis_score += 5
        
        # Pattern-based detection
        if re.search(r'\b(a|an)\s+\w+\s+(of|with|in|on)', prompt_lower):
            generation_score += 2
        
        if re.search(r'(this|that)\s+(image|picture|photo)', prompt_lower):
            analysis_score += 3
        
        # Question patterns
        if prompt_lower.startswith(('what', 'how', 'why', 'where', 'when', 'who')):
            analysis_score += 2
        
        # Command patterns
        if prompt_lower.startswith(('create', 'make', 'generate', 'draw', 'design')):
            generation_score += 3
        
        # Determine action based on scores
        if analysis_score > generation_score:
            action = 'analyze'
            confidence = min(analysis_score / (analysis_score + generation_score + 1), 0.95)
        elif generation_score > analysis_score:
            action = 'generate'
            confidence = min(generation_score / (analysis_score + generation_score + 1), 0.95)
        else:
            action = 'unclear'
            confidence = 0.5
        
        extracted_prompt = self._extract_core_prompt(prompt, action)
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': f"Generation: {generation_score}, Analysis: {analysis_score}, Has image: {has_image_attachment}",
            'extracted_prompt': extracted_prompt
        }
    
    def _extract_core_prompt(self, prompt: str, action: str) -> str:
        """Extract the core prompt by removing command words"""
        prompt_clean = prompt.strip()
        
        if action == 'generate':
            for keyword in sorted(self.generation_keywords, key=len, reverse=True):
                pattern = rf'\b{re.escape(keyword)}\b\s*'
                if prompt_clean.lower().startswith(keyword):
                    prompt_clean = re.sub(pattern, '', prompt_clean, count=1, flags=re.IGNORECASE)
                    break
        
        elif action == 'analyze':
            for keyword in sorted(self.analysis_keywords, key=len, reverse=True):
                pattern = rf'\b{re.escape(keyword)}\b\s*'
                if prompt_clean.lower().startswith(keyword):
                    prompt_clean = re.sub(pattern, '', prompt_clean, count=1, flags=re.IGNORECASE)
                    break
        
        return prompt_clean.strip()

# --- Utility Functions ---
def setup_directories(output_dir: str) -> bool:
    """Create necessary directories if they don't exist"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error creating directory {output_dir}: {e}")
        return False

def save_analysis_results(output_dir: str, image_filename: str, analysis_result: str) -> str:
    """Save image analysis results to a text file"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        base_name = os.path.splitext(image_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(analysis_result)
            
        return output_file
    except Exception as e:
        raise Exception(f"Error saving analysis results: {e}")

# --- Main Image Agent Class ---
class IntelligentImageAgent:
    """Intelligent Image Agent for FastAPI"""
    
    def __init__(self, config: ImageAgentConfig):
        self.config = config
        self.classifier = PromptClassifier()
        self.is_initialized = False
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize Google Cloud services"""
        try:
            logger.info(f"Initializing Vertex AI for project '{self.config.GCP_PROJECT_ID}'...")
            vertexai.init(project=self.config.GCP_PROJECT_ID, location=self.config.GCP_LOCATION)
            
            logger.info("Loading Vertex AI models...")
            self.generation_model = ImageGenerationModel.from_pretrained(self.config.IMAGEN_MODEL)
            self.vision_model = GenerativeModel(self.config.VISION_MODEL)
            logger.info("Vertex AI models initialized successfully.")
            
            # Setup directories
            setup_directories(self.config.OUTPUT_DIR)
            setup_directories(self.config.TEXT_OUTPUT_DIR)
            setup_directories(self.config.UPLOAD_DIR)
            
            self.is_initialized = True
            logger.info("Intelligent Image Agent initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing Image Agent: {e}")
            self.is_initialized = False

    def _is_prompt_educational(self, prompt: str) -> bool:
        """Validate if a prompt is educational and student-friendly"""
        educational_keywords = {
            'solar system', 'planet', 'galaxy', 'atom', 'cell', 'dna', 'molecule',
            'photosynthesis', 'water cycle', 'carbon cycle', 'digestive system',
            'human body', 'skeleton', 'muscle', 'brain', 'heart', 'lung',
            'mathematics', 'geometry', 'algebra', 'fraction', 'equation',
            'historical', 'geography', 'map', 'continent', 'climate',
            'volcano', 'earthquake', 'weather', 'ecosystem', 'diagram', 'chart',
            'artificial intelligence', 'machine learning', 'deep learning',
            'data science', 'algorithm', 'model', 'neural network'
        }
        
        prompt_lower = prompt.lower()
        for keyword in educational_keywords:
            if keyword in prompt_lower:
                logger.info("✅ Automatically validated as educational content")
                return True

        logger.info("🧠 Validating prompt for educational content...")
        
        validation_prompt = f"""Task: Determine if this image generation request is educational.
        
        Request: '{prompt}'
        
        Rules:
        - Must relate to academic subjects (science, math, history, biology, coding etc.)
        - Must be suitable for students (grade school to college)
        - Must have learning value
        - Must be safe and appropriate for education
        
        Respond with only 'YES' or 'NO'.
        """
        
        try:
            response = self.vision_model.generate_content(validation_prompt)
            decision = response.text.strip().upper()
            logger.info(f"Validation decision: {decision}")
            return "YES" in decision
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            if any(word in prompt_lower for word in ['science', 'math', 'biology', 'chemistry', 'physics', 'history', 'geography']):
                logger.info("✅ Approved as likely educational content despite validation error")
                return True
            return False

    async def process_user_request(self, 
                                   prompt: str, 
                                   uploaded_file: Optional[UploadFile] = None) -> Dict:
        """Main method to process user requests with automatic action detection"""
        if not self.is_initialized:
            return {"status": "error", "message": "Agent not properly initialized"}
        
        try:
            has_attachment = uploaded_file is not None
            classification = self.classifier.classify_prompt(prompt, has_attachment)
            
            logger.info(f"🧠 Prompt Classification: {classification['action']} (confidence: {classification['confidence']:.2f})")
            
            if classification['action'] == 'generate':
                result = await self._generate_educational_image(classification['extracted_prompt'])
                
            elif classification['action'] == 'analyze':
                if not uploaded_file:
                    return {
                        "status": "error",
                        "action": "analyze",
                        "message": "Image analysis requested but no image uploaded. Please upload an image file."
                    }
                result = await self._analyze_uploaded_image(uploaded_file, classification['extracted_prompt'])
                
            elif classification['action'] == 'unclear':
                return self._handle_unclear_prompt(prompt, classification)
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {classification['action']}"
                }
            
            result['classification'] = classification
            return result
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing request: {str(e)}"
            }

    async def _generate_educational_image(self, prompt: str, num_images: int = 1) -> Dict:
        """Generate educational images using Vertex AI after validation"""
        
        if not self._is_prompt_educational(prompt):
            return {
                "status": "rejected",
                "action": "generate",
                "message": "Sorry, I can only generate images for educational topics. Please provide a prompt related to learning."
            }
            
        try:
            logger.info(f"🎨 Generating educational image with prompt: '{prompt}'")
            
            enhanced_prompt = (
                "Educational illustration for students, clear and simple vector graphic, "
                "user-friendly, vibrant colors, easy to understand. "
                f"Create an image of: {prompt}"
            )
            
            images = self.generation_model.generate_images(
                prompt=enhanced_prompt,
                number_of_images=num_images,
                aspect_ratio=self.config.DEFAULT_ASPECT_RATIO,
                safety_filter_level=self.config.DEFAULT_SAFETY_LEVEL,
                add_watermark=False,
            )
            
            saved_images = []
            for i, image in enumerate(images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}_{i+1}.png"
                file_path = os.path.join(self.config.OUTPUT_DIR, filename)
                
                image.save(location=file_path, include_generation_parameters=False)
                saved_images.append({
                    "filename": filename,
                    "path": file_path,
                    "url": f"/static/images/{filename}"
                })
                logger.info(f"✅ Successfully saved: {file_path}")
            
            return {
                "status": "success",
                "action": "generate",
                "message": f"Successfully generated {len(saved_images)} educational image(s)",
                "data": {
                    "images": saved_images,
                    "prompt_used": enhanced_prompt,
                    "original_prompt": prompt
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return {
                "status": "error",
                "action": "generate",
                "message": f"Error generating image: {str(e)}"
            }

    async def _analyze_uploaded_image(self, uploaded_file: UploadFile, context_prompt: str = "") -> Dict:
        """Analyze uploaded images using Gemini-2.5-Pro"""
        
        # Validate file
        if not uploaded_file.content_type.startswith('image/'):
            return {
                "status": "error",
                "action": "analyze",
                "message": "Uploaded file is not an image"
            }
        
        try:
            # Read uploaded file
            file_content = await uploaded_file.read()
            
            # Save uploaded file temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"upload_{timestamp}_{uploaded_file.filename}"
            temp_path = os.path.join(self.config.UPLOAD_DIR, temp_filename)
            
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file_content)
            
            logger.info(f"🔍 Analyzing uploaded image: {uploaded_file.filename}")
            
            # Create a Part object for the image
            image_part = Part.from_data(
                mime_type=uploaded_file.content_type,
                data=file_content
            )
            
            # Construct the analysis prompt
            if context_prompt:
                prompt = (
                    f"Analyze the following image with respect to this context: {context_prompt}. "
                    "1. Provide a detailed description of all visual elements in the image, including objects, people, animals, text, colors, and any other distinct features. "
                    "2. Explain the content of the image in detail, including its educational significance, context, or purpose. "
                    "Ensure the explanation is suitable for students and highlights learning value."
                )
            else:
                prompt = (
                    "1. Provide a detailed description of all visual elements in the image, including objects, people, animals, text, colors, and any other distinct features. "
                    "2. Explain the content of the image in detail, including its educational significance, context, or purpose. "
                    "Ensure the explanation is suitable for students and highlights learning value."
                )
            
            # Use Gemini-2.5-Pro for analysis
            response = self.vision_model.generate_content([prompt, image_part])
            description = response.text.strip()
            
            # Save analysis results
            analysis_file = save_analysis_results(
                self.config.TEXT_OUTPUT_DIR, 
                uploaded_file.filename, 
                description
            )
            
            result = {
                "status": "success",
                "action": "analyze",
                "message": "Image analysis completed successfully",
                "data": {
                    "filename": uploaded_file.filename,
                    "uploaded_path": temp_path,
                    "analysis": description,
                    "analysis_file": analysis_file,
                    "context_prompt": context_prompt
                }
            }
            
            logger.info(f"✅ Analysis complete for: {uploaded_file.filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "status": "error",
                "action": "analyze",
                "message": f"Error analyzing image: {str(e)}"
            }

    def _handle_unclear_prompt(self, user_prompt: str, classification: Dict) -> Dict:
        """Handle prompts where the intent is unclear"""
        suggestions = [
            "To generate an image, try: 'Create a diagram of the water cycle'",
            "To analyze an image, upload an image and try: 'Analyze this image' or 'What do you see?'"
        ]
        
        return {
            "status": "clarification_needed",
            "action": "unclear",
            "message": "I'm not sure what you'd like me to do. Could you please clarify?",
            "data": {
                "suggestions": suggestions,
                "classification": classification,
                "original_prompt": user_prompt
            }
        }

# --- FastAPI Application ---
app = FastAPI(
    title="Intelligent Image Agent API",
    description="AI-powered educational image generation and analysis service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
config = ImageAgentConfig()
agent = IntelligentImageAgent(config)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Intelligent Image Agent API...")
    if not agent.is_initialized:
        logger.error("❌ Agent initialization failed!")
    else:
        logger.info("✅ Agent initialized successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Intelligent Image Agent API is running!",
        "status": "healthy" if agent.is_initialized else "unhealthy",
        "version": "1.0.0"
    }

@app.post("/process")
async def process_request(
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Main endpoint to process user requests.
    Automatically determines whether to generate or analyze based on prompt and file upload.
    """
    try:
        result = await agent.process_user_request(prompt, file)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in process_request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )

@app.post("/generate")
async def generate_image(request: GenerateImageRequest):
    """
    Dedicated endpoint for image generation.
    """
    try:
        result = await agent._generate_educational_image(request.prompt, request.num_images)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error generating image: {str(e)}"
            }
        )

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form("")
):
    """
    Dedicated endpoint for image analysis.
    """
    try:
        result = await agent._analyze_uploaded_image(file, prompt)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error analyzing image: {str(e)}"
            }
        )

@app.get("/classify")
async def classify_prompt(prompt: str):
    """
    Endpoint to test prompt classification without processing.
    """
    try:
        classification = agent.classifier.classify_prompt(prompt, False)
        return JSONResponse(content={
            "status": "success",
            "classification": classification
        })
    
    except Exception as e:
        logger.error(f"Error in classify_prompt: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error classifying prompt: {str(e)}"
            }
        )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if agent.is_initialized else "unhealthy",
        "agent_initialized": agent.is_initialized,
        "timestamp": datetime.now().isoformat(),
        "directories": {
            "output": os.path.exists(config.OUTPUT_DIR),
            "text_output": os.path.exists(config.TEXT_OUTPUT_DIR),
            "upload": os.path.exists(config.UPLOAD_DIR)
        }
    }

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="127.0.0.1", port=8080)