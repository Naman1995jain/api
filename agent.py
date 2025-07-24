import os
import sys
import subprocess
import base64
import io
import re
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image as PILImage
import google.generativeai as genai
from datetime import datetime
import vertexai
from vertexai.preview.vision_models import Image, ImageGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part

class ImageAgentConfig:
    """Configuration class for the Image Agent"""
    
    def __init__(self):
        # Google Cloud Project Configuration
        self.GCP_PROJECT_ID = "vertex-ai-466616"  # Replace with your project ID
        self.GCP_LOCATION = "us-central1"
        
        # Image Generation Settings
        self.OUTPUT_DIR = "C:/Vertex/image"  # Directory to save generated images
        self.TEXT_OUTPUT_DIR = "C:/Vertex/image_text"  # Directory to save analysis results
        self.DEFAULT_ASPECT_RATIO = "16:9"
        self.DEFAULT_SAFETY_LEVEL = "block_few"
        
        # Vertex AI Models
        self.IMAGEN_MODEL = "imagen-4.0-generate-preview-06-06"  # For image generation
        self.VISION_MODEL = "gemini-2.5-pro"  # For vision and understanding

# --- Prompt Understanding and Classification ---
class PromptClassifier:
    """Intelligent prompt classification to determine the required action"""
    
    def __init__(self):
        # Keywords for image generation
        self.generation_keywords = {
            'create', 'generate', 'make', 'draw', 'design', 'paint', 'sketch', 
            'produce', 'build', 'craft', 'compose', 'render', 'illustrate',
            'visualize', 'imagine', 'picture', 'show me', 'create an image',
            'generate image', 'make picture', 'draw me', 'design a', 'paint a'
        }
        
        # Keywords for image analysis
        self.analysis_keywords = {
            'analyze', 'describe', 'explain', 'what is', 'what are', 'tell me about',
            'identify', 'recognize', 'detect', 'find', 'look at', 'examine',
            'inspect', 'review', 'understand', 'see', 'observe', 'read',
            'what do you see', 'describe this', 'analyze image', 'what is in',
            'caption', 'summarize', 'interpret', 'decode'
        }
        
        # File extensions that indicate image files
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # Common image-related path indicators
        self.image_path_indicators = {
            'image', 'img', 'photo', 'picture', 'pic', 'screenshot', 'scan'
        }
    
    def classify_prompt(self, prompt: str, has_image_attachment: bool = False) -> Dict:
        """
        Classify the user's prompt to determine the required action
        
        Returns:
        {
            'action': 'generate' | 'analyze' | 'unclear',
            'confidence': float,
            'reasoning': str,
            'extracted_prompt': str
        }
        """
        prompt_lower = prompt.lower().strip()
        
        # Check if there's an image attachment or path mentioned
        image_path = self._extract_image_path(prompt)
        has_image = has_image_attachment or (image_path is not None)
        
        # Score for generation vs analysis
        generation_score = 0
        analysis_score = 0
        
        # Check for explicit keywords
        for keyword in self.generation_keywords:
            if keyword in prompt_lower:
                generation_score += 2
                if keyword in prompt_lower[:20]:  # Early in prompt = higher weight
                    generation_score += 1
        
        for keyword in self.analysis_keywords:
            if keyword in prompt_lower:
                analysis_score += 2
                if keyword in prompt_lower[:20]:  # Early in prompt = higher weight
                    analysis_score += 1
        
        # Strong indicators based on context
        if has_image:
            analysis_score += 5  # Strong bias toward analysis if image present
        
        # Pattern-based detection
        if re.search(r'\b(a|an)\s+\w+\s+(of|with|in|on)', prompt_lower):
            generation_score += 2  # "a picture of", "an image with"
        
        if re.search(r'(this|that)\s+(image|picture|photo)', prompt_lower):
            analysis_score += 3  # "this image", "that picture"
        
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
        
        # Extract the core prompt (remove command words)
        extracted_prompt = self._extract_core_prompt(prompt, action)
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': self._build_reasoning(generation_score, analysis_score, has_image),
            'extracted_prompt': extracted_prompt,
            'image_path': image_path
        }
    
    def _extract_image_path(self, prompt: str) -> Optional[str]:
        """Extract image file path from prompt if present"""
        words = prompt.split()
        for word in words:
            if any(word.lower().endswith(ext) for ext in self.image_extensions):
                path = word.strip('"\'')
                return path if os.path.exists(path) else None
        
        path_pattern = r'[A-Za-z]:[\\\/][\w\\\/\.\-\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp)'
        match = re.search(path_pattern, prompt, re.IGNORECASE)
        if match:
            path = match.group(0)
            return path if os.path.exists(path) else None
        
        return None
    
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
    
    def _build_reasoning(self, gen_score: int, ana_score: int, has_image: bool) -> str:
        """Build human-readable reasoning for the classification"""
        reasons = []
        
        if has_image:
            reasons.append("Image file detected")
        if gen_score > 0:
            reasons.append(f"Generation keywords found (score: {gen_score})")
        if ana_score > 0:
            reasons.append(f"Analysis keywords found (score: {ana_score})")
        
        if not reasons:
            reasons.append("No clear indicators found")
        
        return "; ".join(reasons)

# --- Utility Functions ---
def setup_directories(output_dir: str) -> bool:
    """Create necessary directories if they don't exist"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        return True
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        return False

def display_image(file_path: str) -> None:
    """Opens the specified image file using the default system viewer"""
    print(f"Displaying image: {file_path}")
    try:
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        else:  # linux
            subprocess.run(["xdg-open", file_path], check=True)
    except Exception as e:
        print(f"Error displaying image: {e}")
        print("You can find the saved image in the output directory.")

def save_analysis_results(output_dir: str, image_path: str, analysis_result: Dict) -> str:
    """Save only the image description to a text file"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.txt")
        
        content = analysis_result['description']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return output_file
    except Exception as e:
        raise Exception(f"Error saving analysis results: {e}")

# --- Intelligent Image Agent Class ---
class IntelligentImageAgent:
    """Intelligent Image Agent specialized for educational content."""
    
    def __init__(self, config: ImageAgentConfig):
        self.config = config
        self.classifier = PromptClassifier()
        self.is_initialized = False
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize Google Cloud services"""
        try:
            print(f"Initializing Vertex AI for project '{self.config.GCP_PROJECT_ID}'...")
            vertexai.init(project=self.config.GCP_PROJECT_ID, location=self.config.GCP_LOCATION)
            
            print("Loading Vertex AI models...")
            self.generation_model = ImageGenerationModel.from_pretrained(self.config.IMAGEN_MODEL)
            self.vision_model = GenerativeModel(self.config.VISION_MODEL)  # Use Gemini-2.5-Pro
            print("Vertex AI models initialized successfully.")
            
            setup_directories(self.config.OUTPUT_DIR)
            setup_directories(self.config.TEXT_OUTPUT_DIR)
            
            self.is_initialized = True
            print("Intelligent Image Agent initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing Image Agent: {e}")
            self.is_initialized = False

    def _is_prompt_educational(self, prompt: str) -> bool:
        """Uses Vertex AI to validate if a prompt is educational and student-friendly."""
        educational_keywords = {
            'solar system', 'planet', 'galaxy', 'atom', 'cell', 'dna', 'molecule','photosynthesis', 'water cycle', 'carbon cycle', 'digestive system','human body', 'skeleton', 'muscle', 'brain', 'heart', 'lung',
            'mathematics', 'geometry', 'algebra', 'fraction', 'equation','historical', 'geography', 'map', 'continent', 'climate','volcano', 'earthquake', 'weather', 'ecosystem', 'diagram', 'chart','artificial intelligence', 
            'machine learning', 'deep learning', 'data science','algorithm', 'model', 'neural network', 'large language model (LLM)','supervised learning', 'unsupervised learning', 'reinforcement learning','regression', 'classification', 
            'clustering', 'decision tree', 'random forest','gradient boosting', 'support vector machine (SVM)', 'k-means','convolutional neural network (CNN)', 'recurrent neural network (RNN)', 'transformer','generative adversarial network (GAN)', 'natural language processing (NLP)','computer vision', 'data', 'dataset', 'big data', 'feature', 'label','training', 'testing', 'validation', 'overfitting', 'underfitting','loss function', 'gradient descent', 'backpropagation', 'hyperparameter','python', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy','API', 'cloud computing', 'statistics', 'probability', 'linear algebra', 'calculus'
        }
        
        prompt_lower = prompt.lower()
        for keyword in educational_keywords:
            if keyword in prompt_lower:
                print("✅ Automatically validated as educational content")
                return True

        print("🧠 Validating prompt for educational content...")
        
        validation_prompt = f"""Task: Determine if this image generation request is educational.
        
        Request: '{prompt}'
        
        Rules:
        - Must relate to academic subjects (science, math, history, biological, coding etc.)
        - Must be suitable for students (grade school to college)
        - Must have learning value
        - Must be safe and appropriate for education
        
        Respond with only 'YES' or 'NO'.
        """
        
        try:
            response = self.vision_model.generate_content(validation_prompt)
            decision = response.text.strip().upper()
            print(f"   Validation decision: {decision}")
            return "YES" in decision
            
        except Exception as e:
            print(f"   Error during validation: {e}")
            if any(word in prompt_lower for word in ['science', 'math', 'biology', 'chemistry', 'physics', 'history', 'geography']):
                print("✅ Approved as likely educational content despite validation error")
                return True
            return False

    def process_user_prompt(self, 
                            user_prompt: str, 
                            image_attachment: Optional[str] = None,
                            show_classification: bool = True) -> Dict:
        """
        Main method: Process user prompt and automatically choose the right action
        """
        if not self.is_initialized:
            return {"status": "error", "message": "Agent not properly initialized"}
        
        try:
            has_attachment = image_attachment is not None
            classification = self.classifier.classify_prompt(user_prompt, has_attachment)
            
            if show_classification:
                print(f"🧠 Prompt Classification:")
                print(f"   Action: {classification['action']}")
                print(f"   Confidence: {classification['confidence']:.2f}")
                print(f"   Reasoning: {classification['reasoning']}")
                print(f"   Extracted Prompt: '{classification['extracted_prompt']}'")
                print()
            
            image_path = image_attachment or classification.get('image_path')
            
            if classification['action'] == 'generate':
                result = self._generate_educational_image(classification['extracted_prompt'])
                
            elif classification['action'] == 'analyze':
                if not image_path:
                    return {
                        "status": "error",
                        "message": "Image analysis requested but no image provided. Please provide an image file path or attachment.",
                        "classification": classification
                    }
                result = self._analyze_image(image_path, classification['extracted_prompt'])
                
            elif classification['action'] == 'unclear':
                return self._handle_unclear_prompt(user_prompt, classification)
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {classification['action']}",
                    "classification": classification
                }
            
            result['classification'] = classification
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing prompt: {str(e)}"
            }

    def _generate_educational_image(self, prompt: str, num_images: int = 1) -> Dict:
        """Generate educational images using Vertex AI after validation."""
        
        if not self._is_prompt_educational(prompt):
            return {
                "status": "rejected",
                "action": "generate",
                "message": "Sorry, I can only generate images for educational topics. Please provide a prompt related to learning."
            }
            
        try:
            print(f"🎨 Generating educational image with prompt: '{prompt}'")
            
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
                saved_images.append(file_path)
                print(f"✅ Successfully saved: {file_path}")
                
                if i == 0:
                    display_image(file_path)
            
            return {
                "status": "success",
                "action": "generate",
                "message": f"Successfully generated {len(saved_images)} educational image(s)",
                "images": saved_images,
                "prompt_used": enhanced_prompt
            }
            
        except Exception as e:
            return {
                "status": "error",
                "action": "generate",
                "message": f"Error generating image: {str(e)}"
            }

    def _analyze_image(self, image_path: str, context_prompt: str = "") -> Dict:
        """Analyze images using Gemini-2.5-Pro and provide detailed description and explanation."""
        # Validate image path
        if not os.path.exists(image_path):
            return {
                "status": "error",
                "action": "analyze",
                "message": "Image file not found. Please provide a valid path to your image file."
            }
        
        # Validate image file type
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in valid_extensions:
            return {
                "status": "error",
                "action": "analyze",
                "message": f"Unsupported image format. Please provide an image in one of these formats: {', '.join(valid_extensions)}"
            }
        
        try:
            print(f"🔍 Analyzing image: {image_path}")
            
            # Read the image file as bytes
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Create a Part object for the image
            image_part = Part.from_data(
                mime_type="image/png",  # Adjust MIME type based on image format
                data=image_data
            )
            
            # Construct the prompt for detailed description and explanation
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
            
            result = {
                "status": "success",
                "action": "analyze",
                "image_path": image_path,
                "description": description,
                "prompt_used": prompt
            }
            
            analysis_data = {
                "analysis_type": "intelligent_analysis",
                "description": description,
                "prompt_used": prompt
            }
            output_file = save_analysis_results(self.config.TEXT_OUTPUT_DIR, image_path, analysis_data)
            result["output_file"] = output_file
            
            # Clean console output
            print("-" * 50)
            print("Image Analysis:")
            print(description)
            print("-" * 50)
            print(f"✅ Analysis complete. (Full text also saved to: {output_file})")
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "action": "analyze",
                "message": f"Error analyzing image: {str(e)}"
            }

    def _handle_unclear_prompt(self, user_prompt: str, classification: Dict) -> Dict:
        """Handle prompts where the intent is unclear"""
        suggestions = []
        
        if any(word in user_prompt.lower() for word in ['image', 'picture', 'photo']):
            suggestions.append("To generate an image, try: 'Create a diagram of the water cycle'")
            suggestions.append("To analyze an image, try: 'Analyze [image_path]' or attach an image")
        
        return {
            "status": "clarification_needed",
            "message": "I'm not sure what you'd like me to do. Could you please clarify?",
            "suggestions": suggestions,
            "classification": classification,
            "original_prompt": user_prompt
        }

# --- Interactive Interface ---
class SmartImageInterface:
    """Smart command-line interface for the Intelligent Image Agent"""
    
    def __init__(self):
        self.config = ImageAgentConfig()
        self.agent = IntelligentImageAgent(self.config)
        
    def run_interactive(self):
        """Run the interactive command-line interface"""
        print("=" * 70)
        print("🤖 Educational Image Agent - Natural Language Interface")
        print("=" * 70)
        print("I can create simple, educational images for students.")
        print()
        print("Examples:")
        print("  • 'Create a diagram of the water cycle'")
        print("  • 'Generate a simple drawing of the solar system'") 
        print("  • 'Analyze this image: [drag and drop your image here]'")
        print("  • 'Describe the image: [paste your image path]'")
        print("  • 'What's in this picture: [select your image file]'")
        print()
        print("Commands: 'help', 'quit', 'config'")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\n💬 What would you like me to do? ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                
                elif user_input.lower() == 'config':
                    self._show_config()
                
                elif user_input:
                    self._process_natural_input(user_input)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _process_natural_input(self, user_input: str):
        """Process natural language input and handle results."""
        print(f"\n🤖 Processing: '{user_input}'")
        
        result = self.agent.process_user_prompt(user_input)
        
        if result["status"] == "rejected":
            print(f"⚠️ Request Rejected: {result['message']}")

        elif result["status"] == "clarification_needed":
            print(f"🤔 {result['message']}")
            if result.get('suggestions'):
                print("\n💡 Suggestions:")
                for suggestion in result['suggestions']:
                    print(f"   • {suggestion}")
        
        elif result["status"] == "error":
            print(f"❌ An error occurred: {result['message']}")
    
    def _show_help(self):
        """Show detailed help information"""
        help_text = """
📚 Natural Language Commands:

🎨 Educational Image Generation:
   I will automatically generate an image if your prompt is educational.
   • "Create a beautiful landscape" <-- This will be rejected.
   • "Generate a diagram of a volcano" <-- This will be accepted.
   • "Draw the human heart"
   • "Design a simple chart showing photosynthesis"

🔍 Image Analysis (automatically detected):
   • "Analyze this image: [drag and drop your image here]"
   • "What's in this picture: [select any image from your computer]"
   • "Describe this image: [paste the path to your image]"
   
   Supported image formats: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP

🧠 How it works:
   The agent first validates if your prompt is educational. If it is,
   it enhances the prompt to create a simple, student-friendly graphic.
   Otherwise, it will reject the request.

⚙️ Setup:
   • Set GEMINI_API_KEY environment variable
   • Update GCP_PROJECT_ID in configuration
   • Authenticate: gcloud auth application-default login

🔧 Requirements:
   pip install google-cloud-aiplatform google-generativeai Pillow python-dotenv
        """
        print(help_text)
    
    def _show_config(self):
        """Show current configuration"""
        print("\n⚙️ Current Configuration:")
        print(f"   GCP Project: {self.config.GCP_PROJECT_ID}")
        print(f"   Location: {self.config.GCP_LOCATION}")
        print(f"   Output Directory: {self.config.OUTPUT_DIR}")
        print(f"   Text Output Directory: {self.config.TEXT_OUTPUT_DIR}")
        print(f"   Vision Model: {self.config.VISION_MODEL}")
        print(f"   Imagen Model: {self.config.IMAGEN_MODEL}")
        print(f"   Gemini API Key: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not Set'}")

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting Intelligent Image Agent...")
    
    if len(sys.argv) > 1:
        # Command line mode
        user_prompt = " ".join(sys.argv[1:])
        config = ImageAgentConfig()
        agent = IntelligentImageAgent(config)
        
        result = agent.process_user_prompt(user_prompt)
        
        if result["status"] == "rejected":
            print(f"⚠️ Request Rejected: {result['message']}")
        elif result["status"] == "error":
            print(f"❌ An error occurred: {result['message']}")
    else:
        # Interactive mode
        interface = SmartImageInterface()
        interface.run_interactive()