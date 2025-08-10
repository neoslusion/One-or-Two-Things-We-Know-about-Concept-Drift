#!/usr/bin/env python3
"""
Script to extract text content from the thesis proposal docx file
"""

import docx
import re
import sys

def extract_docx_content(docx_path):
    """Extract and structure content from docx file"""
    try:
        doc = docx.Document(docx_path)
        
        # Store all paragraphs with their formatting info
        content = []
        current_section = None
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
                
            # Check if this is a heading based on style or formatting
            style_name = para.style.name if para.style else ""
            is_bold = any(run.bold for run in para.runs if run.bold is not None)
            
            # Detect headings by style name or bold formatting
            if any(heading_indicator in style_name.lower() for heading_indicator in ['heading', 'title']) or \
               (is_bold and len(text) < 100):
                content.append({
                    'type': 'heading',
                    'text': text,
                    'style': style_name,
                    'level': extract_heading_level(style_name)
                })
            else:
                content.append({
                    'type': 'paragraph',
                    'text': text,
                    'style': style_name
                })
        
        return content
        
    except Exception as e:
        print(f"Error reading docx file: {e}")
        return None

def extract_heading_level(style_name):
    """Extract heading level from style name"""
    if 'heading' in style_name.lower():
        # Try to extract number from heading style
        match = re.search(r'heading\s*(\d+)', style_name.lower())
        if match:
            return int(match.group(1))
    return 1

def organize_content_by_sections(content):
    """Organize content into sections based on headings"""
    sections = {}
    current_section = "Introduction"
    current_content = []
    
    for item in content:
        if item['type'] == 'heading':
            # Save previous section
            if current_content:
                sections[current_section] = current_content
            
            # Start new section
            current_section = item['text']
            current_content = [item]
        else:
            current_content.append(item)
    
    # Save last section
    if current_content:
        sections[current_section] = current_content
    
    return sections

def print_content_structure(sections):
    """Print the structure of extracted content"""
    print("EXTRACTED CONTENT STRUCTURE:")
    print("=" * 50)
    
    for section_name, content in sections.items():
        print(f"\n## {section_name}")
        print("-" * 30)
        
        for item in content:
            if item['type'] == 'heading':
                print(f"HEADING: {item['text']}")
            else:
                # Truncate long paragraphs for structure view
                text = item['text'][:150] + "..." if len(item['text']) > 150 else item['text']
                print(f"PARA: {text}")

def export_to_text_file(sections, output_path):
    """Export organized content to a text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("THESIS PROPOSAL CONTENT EXTRACTION\n")
        f.write("=" * 50 + "\n\n")
        
        for section_name, content in sections.items():
            f.write(f"SECTION: {section_name}\n")
            f.write("-" * 40 + "\n\n")
            
            for item in content:
                if item['type'] == 'heading':
                    f.write(f"### {item['text']}\n\n")
                else:
                    f.write(f"{item['text']}\n\n")
            
            f.write("\n" + "="*40 + "\n\n")

if __name__ == "__main__":
    docx_path = "report/office/Thesis_Proposal.docx"
    output_path = "extracted_proposal_content.txt"
    
    print("Extracting content from thesis proposal...")
    content = extract_docx_content(docx_path)
    
    if content:
        sections = organize_content_by_sections(content)
        print_content_structure(sections)
        export_to_text_file(sections, output_path)
        print(f"\nContent exported to: {output_path}")
        print(f"Found {len(sections)} sections with {len(content)} total items")
    else:
        print("Failed to extract content from docx file")
        sys.exit(1)
