import re

def parse_log(log_path):
    overfull_boxes = []
    current_files = []
    
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            # Track file opening/closing
            file_match = re.search(r'\((\./[^\s\)]+)', line)
            if file_match:
                current_files.append(file_match.group(1))
            
            # Simple heuristic for closing files (might be imprecise but helps)
            if ')' in line and current_files:
                # This is tricky because ) can be inside text. 
                # But usually log lines with ONLY ) are end of files.
                if line.strip() == ')':
                    current_files.pop()

            if "Overfull \\hbox" in line:
                file = current_files[-1] if current_files else "Unknown"
                overfull_boxes.append((file, line.strip()))
                
    return overfull_boxes

if __name__ == "__main__":
    boxes = parse_log("report/latex/main.log")
    for file, msg in boxes:
        print(f"{file}: {msg}")
