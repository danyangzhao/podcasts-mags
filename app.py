import os
import openai
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Configure the OpenAI API key (assuming you've set an environment variable)
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if "podcast" not in request.files:
            return "No file part", 400
        
        file = request.files["podcast"]
        
        if file.filename == "":
            return "No selected file", 400
        
        # Save uploaded file to server (temporary)
        audio_path = os.path.join("uploads", file.filename)
        file.save(audio_path)

        # 1. Transcribe with Whisper
        transcript = transcribe_audio(audio_path)

        # 2. Generate stylized text from transcript
        magazine_article = generate_magazine_style_article(transcript)

        # 3. Optionally, generate relevant images
        images = generate_images_from_text(transcript)

        # 4. Render results
        return render_template("results.html", article=magazine_article, images=images)
    
    return render_template("index.html")

def transcribe_audio(audio_path: str) -> str:
    """
    Send the audio file to the Whisper API for transcription
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def generate_magazine_style_article(transcript: str) -> str:
    """
    Use GPT to transform the transcript into a stylized magazine article
    """
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
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating magazine article: {e}")
        return "Error generating article."

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
    # Create the uploads folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    app.run(debug=True)
