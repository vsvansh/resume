import os
import logging
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from resume_processor import (
    extract_text, preprocess_text, calculate_similarity, extract_skills,
    extract_personal_info, generate_resume_summary, generate_ai_suggestions
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# File upload configuration
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if both resume and job description are provided
    if 'resume' not in request.files:
        flash('No resume file selected', 'danger')
        return redirect(url_for('index'))
    
    job_description = request.form.get('job_description', '').strip()
    if not job_description:
        flash('Please provide a job description', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['resume']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract text from the resume
            resume_text = extract_text(filepath)
            if not resume_text:
                flash('Could not extract text from the resume. Please check the file format.', 'danger')
                return redirect(url_for('index'))
            
            # Extract personal information
            personal_info = extract_personal_info(resume_text)
            
            # Preprocess texts
            processed_resume = preprocess_text(resume_text)
            processed_job = preprocess_text(job_description)
            
            # Calculate similarity
            similarity_score, vectorizer, feature_names = calculate_similarity(processed_resume, processed_job)
            
            # Convert to percentage and round to 2 decimal places
            match_percentage = round(similarity_score * 100, 2)
            
            # Extract skills from job description and resume
            job_skills = extract_skills(job_description)
            resume_skills = extract_skills(resume_text)
            
            # Find missing skills
            missing_skills = [skill for skill in job_skills if skill not in resume_skills]
            
            # Generate resume summary with error handling
            try:
                resume_summary = generate_resume_summary(resume_text, resume_skills)
                # Log the summary for debugging
                logger.debug(f"Successfully generated resume summary with keys: {resume_summary.keys()}")
            except Exception as e:
                logger.error(f"Error generating resume summary: {str(e)}")
                resume_summary = {
                    'experience': [],
                    'education': [],
                    'skills': resume_skills[:5] if resume_skills else [],
                    'general': "Could not generate detailed resume summary."
                }
            
            # Generate AI-based suggestions for improving the resume
            try:
                ai_suggestions = generate_ai_suggestions(resume_text, job_description, missing_skills)
            except Exception as e:
                logger.error(f"Error generating AI suggestions: {str(e)}")
                ai_suggestions = ["Could not generate AI suggestions."]
            
            # Make sure personal info is properly structured
            if not personal_info:
                personal_info = {}
            
            # Store results in session with proper formatting
            session['results'] = {
                'match_percentage': match_percentage,
                'resume_text': resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
                'job_description': job_description[:500] + "..." if len(job_description) > 500 else job_description,
                'missing_skills': missing_skills,
                'resume_skills': resume_skills,
                'job_skills': job_skills,
                'personal_info': personal_info,
                'resume_summary': resume_summary,
                'ai_suggestions': ai_suggestions
            }
            
            # Clean up - remove the uploaded file
            os.remove(filepath)
            
            return redirect(url_for('results'))
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload a PDF or TXT file.', 'danger')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    # Get results from session
    results = session.get('results', None)
    if not results:
        flash('No results found. Please upload a resume first.', 'warning')
        return redirect(url_for('index'))
    
    # Debug information to ensure data is correct
    logger.debug(f"Resume summary: {results.get('resume_summary', {})}")
    logger.debug(f"Personal info: {results.get('personal_info', {})}")
    
    return render_template('results.html', results=results)

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum file size is 16MB.', 'danger')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_server_error(error):
    flash('An unexpected error occurred. Please try again.', 'danger')
    return redirect(url_for('index')), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
