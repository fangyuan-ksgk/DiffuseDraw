import xml.etree.ElementTree as ET
import re, os, glob

def isKanji(v):
	return (v >= 0x4E00 and v <= 0x9FC3) or (v >= 0x3400 and v <= 0x4DBF) or (v >= 0xF900 and v <= 0xFAD9) or (v >= 0x2E80 and v <= 0x2EFF) or (v >= 0x20000 and v <= 0x2A6DF)

def is_valid_kanji_code(code):
    try:
        # Convert hex string code to integer
        kanji_value = int(code, 16)
        return isKanji(kanji_value)
    except ValueError:
        return False
    
def code_to_kanji(code):
    return chr(int(code, 16))

def kanji_to_code(character):
    return f"{ord(character):05x}"

pathre = re.compile(r'<path .*d="([^"]*)".*/>')

def createPathsSVG(f, out_dir: str = "kanji_paths"): # create a new SVG file with black strokes only
	s = open(f, "r", encoding="utf-8").read()
	paths = pathre.findall(s)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(out_dir, f"{os.path.basename(f)[:-4]}-paths.svg")
	out = open(out_path, "w", encoding="utf-8")
	out.write("""<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd" []>
<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109" style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">\n""")
	i = 1
	for path in paths:
		out.write('<!--%2d--><path d="%s"/>\n' % (i, path))
		i += 1
	out.write("</svg>")
	out.close()
	return out_path

def parse_kanjidic2(xml_path):
    """
    Parse KANJIDIC2 XML file and return a dictionary mapping kanji codes to English meanings.
    
    Args:
        xml_path (str): Path to kanjidic2.xml file
    
    Returns:
        dict: {kanji_code: english_meaning} where kanji_code is in "0xxxx" format
    """
    kanji_dict = {}
    
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Iterate through each character entry
    for character in root.findall('character'):
        # Get hex code from codepoint
        cp_elem = character.find('.//cp_value[@cp_type="ucs"]')
        if cp_elem is None:
            continue
            
        # Format code to 5-digit hex with leading zeros
        kanji_code = f"{int(cp_elem.text, 16):05x}"  # Convert to int and back to 5-digit hex
        
        # Get English meanings
        meanings = []
        rm_group = character.find('.//reading_meaning')
        if rm_group is not None:
            rmg = rm_group.find('rmgroup')
            if rmg is not None:
                meanings = [m.text for m in rmg.findall('meaning') 
                          if m.get("xml:lang") is None]
        
        if meanings and is_valid_kanji_code(kanji_code):
            kanji_dict[code_to_kanji(kanji_code)] = '; '.join(meanings)
    
    return kanji_dict

def get_kanji_graphs(kanji_svg_folder: str = "kanji", kanji_graph_folder: str = "kanji_paths"):
    kanji_files = [p for p in glob.glob(f"{kanji_svg_folder}/*.svg") if not "-" in p]
    kanji_codes = [p.split("/")[-1].split(".")[0] for p in kanji_files]

    kanji_graphs = {}
    for file, code in zip(kanji_files, kanji_codes):
        if not is_valid_kanji_code(code):
            continue
        kanji_graphs[code_to_kanji(code)] = createPathsSVG(file, kanji_graph_folder)
    return kanji_graphs


# combine kanji_dict and kanji_meanings
def prepare_kanji_dict(kanji_svg_folder: str = "data/kanji", kanji_graph_folder: str = "data/kanji_paths", kanjidic2_path: str = "data/kanjidic2.xml"):
    kanji_graphs = get_kanji_graphs(kanji_svg_folder, kanji_graph_folder)
    kanji_meanings = parse_kanjidic2(kanjidic2_path)
    
    kanji_dict = {} 
    for kanji, path in kanji_graphs.items():
        if kanji in kanji_meanings:
            kanji_dict[kanji] = {"path": path, "meanings": kanji_meanings[kanji]}
        else:
            kanji_dict[kanji] = {"path": path, "meanings": ""}
    return kanji_dict