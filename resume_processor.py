import re
import PyPDF2
import nltk
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# List of basic personal info keywords to look for
PERSONAL_INFO_KEYWORDS = [
    'name', 'full name', 'phone', 'email', 'address', 'linkedin', 
    'website', 'github', 'location', 'portfolio', 'contact'
]

def extract_text(file_path):
    """Extract text from a PDF or TXT file."""
    try:
        text = ""
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            return None
        
        # Replace multiple whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return None

def preprocess_text(text):
    """Preprocess text for TF-IDF analysis."""
    if not text:
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Use simple splitting as a fallback if NLTK tokenizer fails
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        # Remove stopwords and punctuation
        try:
            stop_words = set(stopwords.words('english'))
        except:
            # Simple fallback stopwords if NLTK fails
            stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                         'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                         'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                         'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                         'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                         'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                         'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                         'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                         'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
                         'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
        
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        
        # Stemming - use simple suffix removal if NLTK stemmer fails
        try:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
        except:
            # Basic fallback stemming (very simplified)
            tokens = [word[:-3] if word.endswith('ing') and len(word) > 5 else word for word in tokens]
            tokens = [word[:-2] if word.endswith('ed') and len(word) > 4 else word for word in tokens]
            tokens = [word[:-1] if word.endswith('s') and len(word) > 3 else word for word in tokens]
        
        # Join tokens back into a string
        processed_text = ' '.join(tokens)
        
        return processed_text
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        # In case of failure, return the original text but lowercased
        return text.lower() if text else ""

def calculate_similarity(resume_text, job_description):
    """Calculate similarity between resume and job description using TF-IDF."""
    try:
        if not resume_text or not job_description:
            return 0.0, None, []
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        
        # Get feature names for potential keyword extraction
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity, vectorizer, feature_names
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0, None, []

def extract_skills(text):
    """
    Extract potential skills from text.
    This is a basic implementation that could be improved with a proper skills database.
    """
    try:
        # List of common technical skills (expanded)
        common_skills = [
            # Programming languages
            'python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust',
            'typescript', 'scala', 'perl', 'r', 'matlab', 'bash', 'shell scripting', 'powershell',
            
            # Web technologies
            'html', 'css', 'sass', 'less', 'react', 'angular', 'vue', 'jquery', 'bootstrap', 'tailwind',
            'node', 'express', 'django', 'flask', 'laravel', 'spring', 'asp.net', 'mvc', 'rest api',
            'graphql', 'json', 'xml', 'websocket', 'oauth', 'pwa', 'spa', 'webpack', 'babel',
            
            # Databases
            'sql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'firebase', 'dynamodb', 'cassandra',
            'redis', 'elasticsearch', 'neo4j', 'mariadb', 'sqlite', 'couchdb', 'nosql', 'rdbms',
            
            # Cloud platforms
            'aws', 'azure', 'gcp', 'cloud computing', 'serverless', 'lambda', 'ec2', 's3', 'rds',
            'heroku', 'digitalocean', 'vercel', 'netlify', 'firebase', 'cloudflare',
            
            # DevOps
            'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'ci/cd', 'github actions',
            'travis ci', 'circle ci', 'monitoring', 'logging', 'prometheus', 'grafana', 'elk stack',
            
            # Version control
            'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'version control',
            
            # Project management
            'agile', 'scrum', 'kanban', 'jira', 'trello', 'asana', 'confluence', 'microsoft project',
            'product management', 'project planning', 'sprint planning', 'sdlc',
            
            # AI/ML
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai', 'artificial intelligence',
            'neural networks', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'data analysis', 'data science', 'big data', 'data visualization', 'etl', 'business intelligence',
            
            # Testing
            'selenium', 'junit', 'pytest', 'jest', 'mocha', 'chai', 'cypress', 'testng', 'qa',
            'quality assurance', 'test automation', 'unit testing', 'integration testing', 'tdd', 'bdd',
            
            # Roles and domains
            'frontend', 'backend', 'full stack', 'devops', 'ui/ux', 'mobile development', 'web development',
            'system administration', 'network administration', 'cybersecurity', 'information security',
            'cloud architecture', 'database administration', 'game development',
            
            # Office and productivity
            'microsoft office', 'excel', 'word', 'powerpoint', 'outlook', 'google workspace', 'sheets',
            'docs', 'tableau', 'power bi', 'looker', 'notion', 'figma', 'adobe photoshop', 'illustrator',
            
            # Soft skills
            'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
            'time management', 'project management', 'analytical', 'creativity', 'adaptability',
            'negotiation', 'presentation', 'organizational', 'detail oriented', 'multitasking',
            'strategic planning', 'decision making', 'conflict resolution', 'coaching', 'mentoring'
        ]
        
        # Also include common single-word skills that might be capitalized in text
        single_word_skills = [
            'Excel', 'Word', 'PowerPoint', 'Outlook', 'Figma', 'Photoshop', 'Illustrator',
            'Kubernetes', 'Docker', 'Terraform', 'Jenkins', 'GitHub', 'GitLab', 'AWS', 'Azure',
            'Python', 'JavaScript', 'TypeScript', 'Java', 'React', 'Angular', 'Vue',
            'SQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Firebase', 'Redis', 'GraphQL',
            'Agile', 'Scrum', 'Kanban', 'Jira', 'Tableau', 'PowerBI', 'Salesforce'
        ]
        
        # Lowercase the text
        text_lower = text.lower()
        
        # Find skills from the common skills list (case-insensitive)
        found_skills = [skill for skill in common_skills if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)]
        
        # Look for skills section in resume
        lines = text.split('\n')
        skills_section = False
        skills_section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            
            # Detect skills section
            if re.search(r'\b(skills|technical skills|core competencies|proficiencies)\b', line_lower) and len(line) < 50:
                skills_section = True
                continue
                
            # End skills section if we hit another section
            if skills_section and re.search(r'\b(experience|education|projects|certifications|references)\b', line_lower) and len(line) < 50:
                skills_section = False
                
            # Collect content from skills section
            if skills_section:
                skills_section_content.append(line)
        
        # Process skills section content
        if skills_section_content:
            # Join all lines in skills section
            skills_text = ' '.join(skills_section_content)
            
            # Extract individual skills by various separators
            potential_skills = []
            
            # Split by common separators
            for separator in [',', '•', '|', '/', ';', '-', '–', '—', '·']:
                if separator in skills_text:
                    items = [item.strip() for item in skills_text.split(separator) if item.strip()]
                    potential_skills.extend(items)
            
            # If no separators found, try splitting by whitespace for single-word skills
            if not potential_skills:
                potential_skills = [word.strip() for word in skills_text.split() if word.strip()]
            
            # Normalize and add to found skills
            for skill in potential_skills:
                # Clean up any remaining punctuation
                skill = re.sub(r'[^\w\s]', '', skill).strip()
                
                if skill and len(skill) > 2 and skill.lower() not in [s.lower() for s in found_skills]:
                    # Check against our list of common capitalized skills
                    if skill in single_word_skills or skill.lower() in [s.lower() for s in common_skills]:
                        found_skills.append(skill.lower())
                    # Otherwise, if it's a reasonable-length word or phrase, add it as a potential skill
                    elif 3 <= len(skill) <= 25 and not re.search(r'\b(the|and|or|with|in|on|at|to|for|a|an|of)\b', skill.lower()):
                        found_skills.append(skill.lower())
        
        # Remove duplicates and sort
        found_skills = list(set(found_skills))
        
        # Prioritize technical skills over soft skills
        technical_skills = [skill for skill in found_skills if skill not in ['leadership', 'communication', 'teamwork', 
                                                                          'problem solving', 'critical thinking',
                                                                          'time management', 'project management']]
        soft_skills = [skill for skill in found_skills if skill in ['leadership', 'communication', 'teamwork', 
                                                                  'problem solving', 'critical thinking',
                                                                  'time management', 'project management']]
        
        # Return technical skills first, then soft skills
        return technical_skills + soft_skills
    except Exception as e:
        logger.error(f"Error extracting skills: {str(e)}")
        return []

def extract_personal_info(text):
    """
    Extract personal information from the resume text.
    """
    # Initialize with empty dict to ensure we always return a valid structure
    personal_info = {
        'name': '',
        'email': '',
        'phone': '',
        'location': '',
        'linkedin': '',
        'github': ''
    }
    
    try:
        if not text:
            logger.warning("Empty text provided for personal information extraction")
            return personal_info
            
        lines = text.split('\n')
        
        # Look for common patterns in the first 20 lines (typically where personal info appears)
        search_lines = min(20, len(lines))
        
        # Basic email pattern - improved to catch various formats
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            personal_info['email'] = email_matches[0]
        
        # Phone number pattern (various formats) - improved to catch more formats
        phone_patterns = [
            r'(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}',  # Standard formats
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',  # (123) 456-7890
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'   # 123-456-7890
        ]
        
        for pattern in phone_patterns:
            phone_matches = re.findall(pattern, text)
            if phone_matches:
                if isinstance(phone_matches[0], tuple):
                    # Handle tuple results from capturing groups
                    phone_str = ''.join(filter(None, phone_matches[0]))
                else:
                    # Handle string results
                    phone_str = phone_matches[0]
                    
                # Format the phone number consistently
                digits_only = re.sub(r'[^\d+]', '', phone_str)
                if len(digits_only) >= 10:  # Only store valid phone numbers
                    if len(digits_only) == 10:  # US number without country code
                        formatted_phone = f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
                    else:
                        formatted_phone = digits_only
                    personal_info['phone'] = formatted_phone
                    break
        
        # Try to find name (usually in the first few lines)
        # Enhanced name detection
        name_found = False
        
        # First, try to find the name using the title/heading pattern in first 10 lines
        for i in range(min(10, search_lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            # Look for a typical resume name pattern (standalone name)
            words = line.split()
            non_name_indicators = ['resume', 'cv', 'curriculum', 'vitae', 'portfolio', 'profile', '@', 'http', 
                                  'summary', 'experience', 'education', 'skills', 'contact', 'career', 'objective']
            
            # Check if this line looks like a name (2-3 words, capitalized, no typical resume section keywords)
            if (2 <= len(words) <= 3 and 
                all(word[0].isupper() for word in words if len(word) > 1) and
                not any(indicator in line.lower() for indicator in non_name_indicators) and
                not re.search(email_pattern, line) and
                not any(re.search(p, line) for p in phone_patterns)):
                personal_info['name'] = line
                name_found = True
                break
        
        # If no name found, try to look for name: pattern or a common first line name
        if not name_found:
            for i in range(min(10, search_lines)):
                line = lines[i].strip()
                if not line:
                    continue
                
                # Try to find "Name: John Smith" pattern
                name_line_match = re.search(r'(?i)(?:^|[^a-z])(name|full name)\s*:\s*([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})', line)
                if name_line_match:
                    personal_info['name'] = name_line_match.group(2)
                    name_found = True
                    break
                
                # As a fallback, check for capitalized first line with 1-4 words
                words = line.split()
                non_name_indicators = ['resume', 'cv', 'curriculum', 'vitae', 'portfolio', 'profile', '@', 'http', 
                                      'summary', 'experience', 'education', 'skills', 'contact', 'career', 'objective']
                if (i <= 2 and 1 <= len(words) <= 4 and 
                    all(word[0].isupper() for word in words if len(word) > 1) and
                    not any(indicator in line.lower() for indicator in non_name_indicators) and
                    not re.search(email_pattern, line) and
                    not any(re.search(p, line) for p in phone_patterns)):
                    personal_info['name'] = line
                    name_found = True
                    break
        
        # Try resume file name to infer name if still not found
        if not name_found and 'John Smith' not in text:  # Avoid picking up sample text
            # Look for a name pattern at start of the document
            name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})'
            name_match = re.search(name_pattern, text.strip())
            if name_match:
                personal_info['name'] = name_match.group(1)
                name_found = True
        
        # LinkedIn profile - improved to handle more URL formats
        linkedin_patterns = [
            r'linkedin\.com/in/[A-Za-z0-9_-]+',
            r'linkedin\.com/profile/[A-Za-z0-9_-]+'
        ]
        
        for pattern in linkedin_patterns:
            linkedin_matches = re.findall(pattern, text.lower())
            if linkedin_matches:
                if not linkedin_matches[0].startswith('http'):
                    personal_info['linkedin'] = "https://www." + linkedin_matches[0]
                else:
                    personal_info['linkedin'] = linkedin_matches[0]
                break
        
        # GitHub profile - improved to handle more URL formats
        github_pattern = r'github\.com/[A-Za-z0-9_-]+'
        github_matches = re.findall(github_pattern, text.lower())
        if github_matches:
            if not github_matches[0].startswith('http'):
                personal_info['github'] = "https://www." + github_matches[0]
            else:
                personal_info['github'] = github_matches[0]
        
        # Try to extract address/location from first 15 lines - improved to handle more formats
        location_found = False
        for i in range(search_lines):
            line = lines[i].strip()
            line_lower = line.lower()
            
            # Look for location indicators or address patterns
            if any(indicator in line_lower for indicator in ['address:', 'location:', 'address', 'location']) and len(line) > 10:
                personal_info['location'] = line
                location_found = True
                break
                
            # Look for potential city, state patterns (e.g., "New York, NY")
            if not location_found and re.search(r'[A-Za-z\s]+,\s*[A-Z]{2}', line):
                personal_info['location'] = line
                location_found = True
                break
                
            # Look for zip code patterns
            if not location_found and re.search(r'[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}', line):
                personal_info['location'] = line
                location_found = True
                break
        
        # Remove None values from the dictionary
        personal_info = {k: v for k, v in personal_info.items() if v is not None}
        
        # If we found anything, log success
        if personal_info:
            logger.info(f"Successfully extracted personal info with keys: {personal_info.keys()}")
        else:
            logger.warning("No personal information could be extracted from the resume")
            
        return personal_info
        
    except Exception as e:
        logger.error(f"Error extracting personal info: {str(e)}")
        # Return the initialized dictionary with None values
        return personal_info

def generate_resume_summary(text, skills):
    """
    Generate a structured summary of the resume content.
    Returns a dictionary with separate sections for experience, education, and skills.
    """
    try:
        # Extract sentences and lines
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        lines = text.split('\n')
        
        # Initialize summary sections
        summary = {
            'experience': [],
            'education': [],
            'skills': skills[:8] if skills else [],  # Include top 8 skills
            'years_experience': None,
            'degree': None,
            'positions': []
        }
        
        # Find potential experience statements and years of experience
        experience_indicators = ['year', 'years', 'experience', 'worked', 'developed', 'managed', 'led', 'created', 'implemented']
        position_titles = ['engineer', 'developer', 'manager', 'director', 'analyst', 'designer', 'architect', 
                           'consultant', 'specialist', 'coordinator', 'administrator', 'lead', 'head', 'chief']
        
        # Try to extract years of experience
        years_pattern = r'(\d+)\+?\s*(?:years?|yrs)(?:\s+of\s+|\s+)(?:experience|work)'
        years_match = re.search(years_pattern, text.lower())
        if years_match:
            summary['years_experience'] = years_match.group(1)
        
        # Extract position titles
        for sentence in sentences:
            lower_sent = sentence.lower()
            for title in position_titles:
                if title in lower_sent:
                    position_match = re.search(r'\b\w+\s+' + re.escape(title) + r'\b', lower_sent)
                    if position_match and position_match.group() not in summary['positions']:
                        summary['positions'].append(position_match.group())
        
        # Look for experience section and extract points
        experience_section = False
        bullet_points = []
        
        # First, try to find an experience section with bullet points
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            
            # Detect beginning of experience section
            if any(indicator in line_lower for indicator in ['experience', 'work history', 'employment', 'professional experience']):
                experience_section = True
                continue
                
            # Detect end of experience section
            if experience_section and any(indicator in line_lower for indicator in 
                                         ['education', 'skills', 'certifications', 'awards', 'references']):
                experience_section = False
                
            # Extract bullet points from experience section
            if experience_section and (line.startswith('•') or line.startswith('-') or 
                                     line.startswith('*') or re.match(r'^\d+\.', line) or
                                     re.match(r'^\[\+\]', line)):
                bullet_points.append(line)
                
            # Also look for lines that begin with action verbs
            if experience_section and len(line.split()) >= 3:
                first_word = line.split()[0].lower()
                action_verbs = ['managed', 'led', 'developed', 'created', 'implemented', 'designed', 
                               'coordinated', 'achieved', 'increased', 'decreased', 'improved', 'negotiated']
                if first_word in action_verbs:
                    bullet_points.append(line)
        
        # If we found bullet points, use them as experience statements
        if bullet_points:
            # Limit to 4 bullet points
            summary['experience'] = bullet_points[:4]
        else:
            # Fallback to sentence-based extraction
            for sentence in sentences:
                lower_sent = sentence.lower()
                if (any(indicator in lower_sent for indicator in experience_indicators) or 
                    re.search(r'\d+\s+year', lower_sent)) and len(sentence) < 200:
                    # Only include reasonably sized sentences
                    summary['experience'].append(sentence)
                    if len(summary['experience']) >= 3:  # Limit to 3 experience statements
                        break
        
        # Get education info with enhanced patterns
        education_indicators = ['education', 'university', 'college', 'bachelor', 'master', 'phd', 'degree', 
                              'graduated', 'certification', 'diploma', 'gpa', 'academic', 'school']
        degree_patterns = [
            r'(bachelor|master|doctorate|ph\.?d\.?|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|m\.?b\.?a\.?)',
            r'(bachelor\'s|master\'s|doctoral|doctorate)(\s+degree)?',
            r'(associate\'s|bachelors|masters)(\s+degree)?'
        ]
        
        # Try to extract degree with more comprehensive regex
        for pattern in degree_patterns:
            degree_match = re.search(pattern, text.lower())
            if degree_match:
                summary['degree'] = degree_match.group(0)
                break
                
        # Look for education section
        education_section = False
        education_points = []
        
        # Try to find education section with structured info
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            
            # Detect beginning of education section
            if any(indicator in line_lower for indicator in education_indicators) and len(line) < 50:
                education_section = True
                continue
                
            # Detect end of education section
            if education_section and any(indicator in line_lower for indicator in 
                                        ['experience', 'skills', 'certifications', 'awards', 'references']):
                education_section = False
                
            # Extract education info
            if education_section:
                # Look for university/college names
                if ('university' in line_lower or 'college' in line_lower or 
                    'institute' in line_lower or 'school' in line_lower) and len(line) < 100:
                    education_points.append(line)
                    
                # Look for degree info
                if any(deg in line_lower for deg in ['bachelor', 'master', 'phd', 'doctorate', 'diploma', 'degree']):
                    education_points.append(line)
                    
                # Look for graduation info
                if ('graduate' in line_lower or 'graduated' in line_lower or 
                    re.search(r'\b20\d\d\b', line) or re.search(r'\b19\d\d\b', line)):
                    education_points.append(line)
                    
                # Check for GPA information
                if 'gpa' in line_lower or 'grade point average' in line_lower:
                    education_points.append(line)
                    
        # If we found education points, use them
        if education_points:
            # Limit to 3 points
            summary['education'] = education_points[:3]
        else:
            # Fallback to sentence-based extraction
            for sentence in sentences:
                lower_sent = sentence.lower()
                if any(indicator in lower_sent for indicator in education_indicators) and len(sentence) < 200:
                    summary['education'].append(sentence)
                    if len(summary['education']) >= 2:  # Limit to 2 education statements
                        break
        
        # If we couldn't find enough structured info, add a fallback general summary
        if not summary['experience'] and not summary['education'] and not summary['skills']:
            # Simple fallback
            summary['general'] = "Could not extract detailed information from this resume."
        
        return summary
    except Exception as e:
        logger.error(f"Error generating resume summary: {str(e)}")
        return {
            'experience': [],
            'education': [],
            'skills': skills[:5] if skills else [],
            'years_experience': None,
            'positions': [],
            'degree': None,
            'general': "Could not generate resume summary due to an error."
        }

def generate_ai_suggestions(resume_text, job_description, missing_skills):
    """
    Generate AI-based suggestions for improving the resume.
    """
    try:
        suggestions = []
        
        # Basic suggestions based on resume length
        if len(resume_text) < 1500:  # Arbitrary length threshold
            suggestions.append("Your resume appears quite brief. Consider adding more details about your experience and achievements.")
        
        # Suggestions based on missing skills
        if missing_skills:
            suggestions.append(f"Add these missing skills to your resume if you have them: {', '.join(missing_skills[:5])}{'...' if len(missing_skills) > 5 else ''}")
            
        # Check for action verbs
        action_verbs = ['achieved', 'improved', 'launched', 'developed', 'created', 'increased', 'decreased', 'resolved', 'managed', 'led']
        resume_lower = resume_text.lower()
        found_verbs = [verb for verb in action_verbs if verb in resume_lower]
        
        if len(found_verbs) < 3:  # Arbitrary threshold
            suggestions.append("Use more action verbs like 'achieved', 'improved', 'launched', etc. to highlight your accomplishments.")
        
        # Check for quantifiable achievements
        if not re.search(r'\d+%|\d+\s+percent', resume_lower) and not re.search(r'\$\d+', resume_lower):
            suggestions.append("Add quantifiable achievements with percentages, numbers, or monetary values to make your impact more concrete.")
        
        # Check for technologies/tools mentioned in job description but not in resume
        job_words = set(job_description.lower().split())
        resume_words = set(resume_lower.split())
        potential_tools = [word for word in job_words if word not in resume_words 
                          and len(word) > 4  # Ignore short words
                          and not any(char in ',.;:()[]{}' for char in word)]  # Ignore words with punctuation
        
        if potential_tools:
            top_tools = potential_tools[:3]
            suggestions.append(f"The job mentions '{', '.join(top_tools)}' which don't appear in your resume. Include these if you have experience with them.")
        
        # Suggestions based on common resume improvements
        suggestions.append("Make sure your accomplishments are specific and tailored to this job description.")
        suggestions.append("Keep your resume concise and focused on relevant experience for this position.")
        
        return suggestions
    except Exception as e:
        logger.error(f"Error generating AI suggestions: {str(e)}")
        return ["Could not generate AI suggestions."]
