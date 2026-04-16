from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

def generate_pdf(question, basic_ans, openai_ans, score_basic, score_openai):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph(f"<b>Question:</b> {question}", styles["Normal"]))
    content.append(Paragraph(f"<b>Basic AI Answer:</b> {basic_ans}", styles["Normal"]))
    content.append(Paragraph(f"<b>OpenAI Answer:</b> {openai_ans}", styles["Normal"]))
    content.append(Paragraph(f"<b>Basic Score:</b> {score_basic}", styles["Normal"]))
    content.append(Paragraph(f"<b>OpenAI Score:</b> {score_openai}", styles["Normal"]))

    doc.build(content)
    return temp_file.name
