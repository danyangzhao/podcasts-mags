import os
import openai
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import threading
import tempfile

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Configure the OpenAI API key (assuming you've set an environment variable)
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Global variables for status and results
processing_status = {
    "status": "",
    "complete": False
}
processing_results = {
    "article": "",
    "images": []
}

@app.route("/progress")
def progress():
    return jsonify(processing_status)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        global processing_status, processing_results
        processing_status = {"status": "Starting process...", "complete": False}
        processing_results = {"article": "", "images": []}
        
        if "podcast" not in request.files:
            return "No file part", 400
        
        file = request.files["podcast"]
        if file.filename == "":
            return "No selected file", 400
        
        # Save the file data
        file_data = file.read()
        
        def process_audio():
            global processing_status, processing_results
            try:
                # Create a temporary file and write the saved data to it
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_data)
                    temp_file.flush()  # Ensure all data is written
                    temp_path = temp_file.name
                
                print(f"Temporary file created at: {temp_path}")  # Debug log
                print(f"File size: {os.path.getsize(temp_path)} bytes")  # Debug log
                
                processing_status["status"] = "Transcribing audio..."
                transcript = transcribe_audio(temp_path)
                
                if not transcript:
                    raise Exception("Transcription returned empty result")
                    
                processing_status["status"] = "Generating magazine article..."
                magazine_article = generate_magazine_style_article(transcript)
                
                processing_status["status"] = "Creating illustrations..."
                images = generate_images_from_text(transcript)
                
                # Store results in global variable
                processing_results["article"] = magazine_article
                processing_results["images"] = images
                
                processing_status["status"] = "Complete!"
                processing_status["complete"] = True
                
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Error cleaning up temporary file: {e}")
                    
            except Exception as e:
                print(f"Error in process_audio: {str(e)}")
                processing_status["status"] = f"Error: {str(e)}"
                processing_status["complete"] = True
                processing_results["article"] = f"An error occurred: {str(e)}"

        # Start processing in background
        thread = threading.Thread(target=process_audio)
        thread.start()
        
        return render_template("processing.html")
    
    return render_template("index.html")

@app.route("/results")
def results():
    global processing_results
    # Store results in session when accessed through request context
    if processing_status["complete"]:
        session['results'] = processing_results
        return render_template("results.html", 
                             article=processing_results["article"],
                             images=processing_results["images"])
    return redirect(url_for('index'))

def transcribe_audio(audio_path: str) -> str:
    """
    Send the audio file to the Whisper API for transcription
    """
    try:
        print(f"Starting transcription of file: {audio_path}")  # Debug log
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            
            if not transcript or not transcript.text:
                raise Exception("Transcription returned empty result")
                
            print(f"Transcription successful. Length: {len(transcript.text)}")  # Debug log
            return transcript.text
            
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")  # Debug log
        raise Exception(f"Transcription failed: {str(e)}")

def generate_magazine_style_article(transcript: str) -> str:
    """
    Use GPT to transform the transcript into a stylized magazine article
    """
    if not transcript:
        return "Error: No transcript was provided for processing."
        
    try:
        prompt = f"""
        You are an expert writer for a popular magazine. Take the following transcript
        and turn it into a well-structured, engaging magazine-style article.
        
        Use proper HTML formatting with these elements:
        - Wrap the title in <h1> tags
        - Use <h2> for section headings
        - Use <p> tags for paragraphs
        - Use <blockquote> for important quotes
        - Break up the text into shorter paragraphs for readability
        
        Ignore the podcast's title and host.
        Ignore advertisements for products or services.
        
        Transcript: {transcript}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,  # Increased token limit
            temperature=0.7
        )
        
        if not response.choices or not response.choices[0].message.content:
            raise Exception("GPT response was empty")
            
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in generate_magazine_style_article: {str(e)}")  # Debug log
        return f"Error generating article: {str(e)}"

def generate_images_from_text(transcript: str) -> list:
    """
    Use DALL·E 3 to generate high-quality images based on the transcript's content.
    """
    prompt_for_image = f"Generate a detailed illustration or photograph relevant to the theme: {transcript[:50]}"

    images = []
    try:
        response = client.images.generate(
            model="dall-e-3",  # Updated to DALL-E 3
            prompt=prompt_for_image,
            n=1,  # DALL-E 3 only supports n=1
            size="1024x1024",  # DALL-E 3 supports: "1024x1024", "1792x1024", or "1024x1792"
            quality="standard",  # Can be "standard" or "hd"
            style="natural"  # Can be "natural" or "vivid"
        )
        # response.data is a list of Image objects with url attribute
        for img_data in response.data:
            images.append(img_data.url)
    except Exception as e:
        print(f"Error generating images: {e}")
    
    return images

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
