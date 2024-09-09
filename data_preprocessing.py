import os
import pandas as pd
import re
import json

data_dir = 'data_summary'
directory = 'data_json'

def parse_case_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
        # text = re.sub(r'[^\w\s]', '', text)

    # Define regex patterns for different sections
    patterns = {
        'Title': r'Case Name\s*(.*)',
        'Key Issues': r'Key Issues\s*:\s*(.*?)(?=\n##|\n\*\*|\n3 Legal Principles Involved|\nLegal Principles Involved|\n3. Legal Principles Involved|\Z)',
        'Relevant Statutes and Provisions': r'Relevant Statutes and Provisions\s*:\s*(.*?)\s*(?=\*\*Citation|\*\*Precedents Cited|\*\*Arguments Presented|\*\*Court’s Analysis and Reasoning|\*\*Judgment|\*\*Implications|\*\*Summary Points|\*\*References|\*\*Further Reading|\s*$)',
        'Precedents Cited': r'Precedents Cited\s*:\s*(.*?)\s*(?=\*\*Citation|\*\*Legal Doctrines|\*\*Arguments Presented|\*\*Court’s Analysis and Reasoning|\*\*Judgment|\*\*Implications|\*\*Summary Points|\*\*References|\*\*Further Reading|\s*$)',
        'Legal Doctrines': r'Legal Doctrines\s*:\s*(.*?)(?=\n##\s*[0-9]+\.\s*[A-Za-z ]+|\Z)',
        'Jurisdiction': r'Court\s*:\s*(.*?)\s*(?=\*\*Citation|\*\*Date of Judgment|\*\*Court’s Analysis and Reasoning|\*\*Judgment|\*\*Implications|\*\*Summary Points|\*\*References|\*\*Further Reading|\s*$)',
        'Date': r'Date of Judgment\s*:\s*(.*?)\s*(?=\*\*Citation|\*\*Background and Context|\*\*Arguments Presented|\*\*Court’s Analysis and Reasoning|\*\*Judgment|\*\*Implications|\*\*Summary Points|\*\*References|\*\*Further Reading|\s*$)'
    }

    # Initialize the dictionary for storing extracted information
    case_data = {
        'Title': None,
        'Summary': text,
        'Full Text': None,
        'Key Issues': '',
        'Relevant Statutes and Provisions': '',
        'Precedents Cited': '',
        'Legal Doctrines': '',
        'Jurisdiction': None,
        'Date': None
    }

    # Extract title
    title_match = re.search(patterns['Title'], text)
    if title_match:
        case_data['Title'] = re.sub(r'[^\w\s]', '', title_match.group(1)).strip()

    key_issues_match = re.search(patterns['Key Issues'], text, re.DOTALL | re.IGNORECASE)
    if key_issues_match:
        case_data['Key Issues'] = re.sub(r'[^\w\s]', '', key_issues_match.group(1)).strip()
        # case_data['Key Issues'] = key_issues_match.group(1).strip()

    # Extract sections
    for key, pattern in patterns.items():
        if key == 'Title' or key == 'Key Issues':
            continue
        key_issues_match = re.search(pattern, text, re.DOTALL)
        if key_issues_match:
            # Clean and join the section content
            section_text = key_issues_match.group(1).strip()
            if key in ['Relevant Statutes and Provisions', 'Precedents Cited', 'Legal Doctrines']:
                case_data[key] = '\n'.join([item.strip() for item in section_text.split('\n') if item.strip()])
            else:
                case_data[key] = re.sub(r'[^\w\s]', '', section_text).strip()

    return case_data

data = []
for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(data_dir, filename)
        case_data = parse_case_file(file_path)
        case_data['Full Text'] = '\n'.join(open(f"data/{filename.split('case_summary')[1]}").read().split('\n')[1:])
        data.append(case_data)

df = pd.DataFrame(data)
df.to_csv('data.csv', index=True, index_label='Case ID')



