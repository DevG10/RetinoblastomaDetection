import streamlit as st
import numpy as np
import os
from PIL import Image
from dotenv import load_dotenv
import pickle
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

CLASS_NAMES = [
    "Bilateral Retinoblastoma",
    "Left Eye Retinoblastoma", 
    "Right Eye Retinoblastoma", 
    "Healthy"
]

def local_css():
    """Apply custom CSS styles to the Streamlit application."""
    st.markdown("""
    <style>
    .main-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    .title {
        color: #1E40AF;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stRadio > div {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .high-confidence {
        color: #10B981;
        font-weight: bold;
    }
    .low-confidence {
        color: #EF4444;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_pdf_report(predictions, image_path):
    """Generate a visually enhanced PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    styles = getSampleStyleSheet()
    report_content = []
    
    # Title and Intro Section
    report_content.append(Paragraph("RetinoNet Diagnostic Report", styles['Title']))
    report_content.append(Spacer(1, 12))
    intro_text = """
    <font size=12>
    Dear Patient,<br/>
    Below is the diagnostic report based on the retinal image analysis. Please note that this is a screening tool and should not replace a professional medical diagnosis.<br/>
    We recommend that you consult a healthcare professional for confirmation and further actions.
    </font>
    """
    report_content.append(Paragraph(intro_text, styles['Normal']))
    report_content.append(Spacer(1, 24))
    
    # Image Section (Display Image of Retinal Scan)
    img = RLImage(image_path, width=300, height=300)
    report_content.append(img)
    report_content.append(Spacer(1, 24))
    
    # Prediction Results Section (Prediction Table)
    report_content.append(Paragraph("<b>Prediction Results:</b>", styles['Heading2']))
    
    prediction_data = [("Class Name", "Confidence Level")]
    for pred in predictions:
        sorted_indices = np.argsort(pred)[::-1]
        
        for idx in sorted_indices:
            confidence = pred[idx]
            class_name = CLASS_NAMES[idx]
            prediction_data.append([class_name, f"{confidence:.2%}"])
    
    # Table to display prediction results
    table = Table(prediction_data, colWidths=[150, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    report_content.append(table)
    report_content.append(Spacer(1, 24))
    
    recommendation = get_recommendation(predictions)
    report_content.append(Paragraph(f"<b>Thoughts:</b>", styles['Heading2']))
    report_content.append(Spacer(1, 12))
    report_content.append(Paragraph(recommendation, styles['Normal']))
    report_content.append(Spacer(1, 24))
    footer_text = """
    <font size=10>
    RetinoNet is a diagnostic tool for educational and screening purposes. Always consult with a certified medical professional for diagnosis and treatment.<br/>
    </font>
    """
    report_content.append(Spacer(1, 24))
    report_content.append(Paragraph(footer_text, styles['Normal']))
    
    doc.build(report_content)
    buffer.seek(0)
    return buffer

def get_recommendation(predictions):
    """Generate personalized recommendations based on predictions."""
    recommendations = {
        "Bilateral Retinoblastoma": "We recommend that you seek immediate consultation with a specialist. Early intervention is crucial.",
        "Left Eye Retinoblastoma": "We suggest visiting an ophthalmologist as soon as possible to discuss treatment options for your left eye.",
        "Right Eye Retinoblastoma": "Immediate consultation with a specialist is recommended to explore potential treatments for your right eye.",
        "Healthy": "Your retinal scan appears normal. However, regular checkups are always encouraged to maintain good eye health."
    }
    
    # Check top prediction and return recommendation
    top_prediction_idx = np.argmax(predictions[0])
    class_name = CLASS_NAMES[top_prediction_idx]
    return recommendations.get(class_name, "No recommendation available.")

def send_email_report(email, pdf_buffer, image):
    """Send email with PDF report attached."""
    try:
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = 'RetinoNet Diagnostic Report'
        
        # Email body
        body = """
        Dear Patient,

        Please find your attached RetinoNet diagnostic report.

        IMPORTANT: This is a screening tool. Always consult a medical professional 
        for a definitive diagnosis.

        Best regards,
        RetinoNet Team
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        pdf_part = MIMEApplication(pdf_buffer.getvalue(), _subtype='pdf')
        pdf_part.add_header('Content-Disposition', 'attachment', filename='RetinoNet_Report.pdf')
        msg.attach(pdf_part)
        
        # Send email with more detailed error handling
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                return True
        
        except smtplib.SMTPAuthenticationError as auth_error:
            st.error(f"Authentication Failed: {auth_error}")
            return False
        except smtplib.SMTPException as smtp_error:
            st.error(f"SMTP Error: {smtp_error}")
            return False
        except Exception as general_error:
            st.error(f"Unexpected Error: {general_error}")
            return False
    
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

@st.cache_resource
def load_model(path):
    """Load the machine learning model from a pickle file."""
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def print_predictions(predictions):
    """Display prediction results with confidence levels."""
    np.set_printoptions(suppress=True)
    
    with st.container():
        st.markdown('<h3 style="color:#1E40AF;">Prediction Results</h3>', unsafe_allow_html=True)
        
        for pred in predictions:
            st.markdown("### Confidence Levels:", unsafe_allow_html=True)
            
            # Sort predictions in descending order
            sorted_indices = np.argsort(pred)[::-1]
            
            for idx in sorted_indices:
                confidence = pred[idx]
                class_name = CLASS_NAMES[idx]
                
                # Color-code confidence levels
                if confidence > 0.7:
                    confidence_class = 'high-confidence'
                    confidence_icon = '✅'
                elif confidence > 0.3:
                    confidence_class = ''
                    confidence_icon = '⚠️'
                else:
                    confidence_class = 'low-confidence'
                    confidence_icon = '❌'
                
                st.markdown(f'<p class="{confidence_class}">{confidence_icon} {class_name}: {confidence:.2%}</p>', 
                            unsafe_allow_html=True)

def main():
    """Main Streamlit application function."""
    # Load environment variables
    load_dotenv()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title='RetinoNet', 
        page_icon=':eye:', 
        layout="wide"
    )
    
    # Apply custom CSS
    local_css()
    
    # Page Title and Subtitle
    st.markdown('<h1 class="title">RetinoNet: Retinal Image Diagnostics 👁️</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered Retinoblastoma Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar Information
    st.sidebar.title("🩺 About RetinoNet")
    st.sidebar.info("""
    ### What is Retinoblastoma?
    Retinoblastoma is a rare type of eye cancer that primarily affects children. 
    Early detection is crucial for successful treatment.
    """)
    st.sidebar.markdown("### ℹ️ Accuracy Notice")
    st.sidebar.warning("""
    This is a screening tool. Always consult medical professionals 
    for definitive diagnosis.
    """)
    
    # Load machine learning model
    try:
        model = load_model(path=os.getenv('MODEL_PATH'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Image Upload Section
    st.markdown("### 📸 Image Upload")
    input_method = st.radio(
        "Image Input Method",
        ("Upload Image", "Camera Capture")
    )
    
    image = None
    
    # Handle image input based on selected method
    if input_method == "Upload Image":
        uploaded_image = st.file_uploader(
            "Upload Retinal Image", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear retinal image for analysis"
        )
        if uploaded_image is not None:
            image = Image.open(uploaded_image).resize((224, 224))
    
    elif input_method == "Camera Capture":
        camera_image = st.camera_input(
            "Capture Retinal Image", 
            help="Take a photo using your device's camera"
        )
        if camera_image is not None:
            image = Image.open(camera_image).resize((224, 224))
    
    # Process and analyze image
    if image is not None:
        _, col2, _ = st.columns([1,3,1])
        with col2:
            st.image(image, caption='Submitted Retinal Image', use_column_width=True)
        
        image_array = np.array(image)
        image_array = image_array / 255.0
        normalized_image = np.expand_dims(image_array, axis=0)
        
        with st.spinner('Analyzing image...'):
            preds = model.predict(normalized_image)
    
        print_predictions(preds)
        
        # Email Report Section
        st.header("Email Report")
        email = st.text_input("Enter your email address:")
        send_report = st.button('Send Report')

        if send_report:
            st.write(f"Email entered: {email}")  # Use st.write instead of print
            
            if email and '@' in email:
                try:
                    # Save temporary image
                    temp_image_path = 'temp_retinal_image.png'
                    image.save(temp_image_path)

                    # Generate PDF
                    st.write("Generating PDF Report...")
                    pdf_buffer = generate_pdf_report(preds, temp_image_path)
                    
                    # Send email
                    result = send_email_report(email, pdf_buffer, image)
                    
                    if result:
                        st.success('Report sent successfully!')
                    else:
                        st.error('Failed to send email. Please check your configuration.')
                    
                    # Remove temporary image
                    os.remove(temp_image_path)
                
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.error('Please enter a valid email address.')
    else:
        st.info("Please upload or capture a retinal image for analysis.")

# Run the main application
if __name__ == "__main__":
    main()