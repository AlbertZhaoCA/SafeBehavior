from utils.data import read_json_from_file,write_json_to_file, get_all_files

def get_all_successful_jailbreaks(path: str = "data/evaluation", save_path: str = "output/successful_jailbreaks_response.json"):
    """
    Get all successful jailbreaks from the dataset.
    """
    count = 0
    successful_jailbreaks = set()
    files = get_all_files(path)
    print(f"Found {len(files)} files in {path}")
    for file in files:
        evaluated = read_json_from_file(file)
        metadata = evaluated[0]["metadata"]
        for eval in evaluated:
            if (evaluation := eval.get("evaluation")) is not None and isinstance(evaluation,dict) and evaluation.get("jailbreak_success") == True:
                count += 1
                successful_jailbreaks.add(eval["response"])

    indexed_jailbreaks = {i: jb for i, jb in enumerate(successful_jailbreaks)}
    write_json_to_file(save_path, indexed_jailbreaks)   

import json

def convert_json_to_readable_text(json_path: str, output_txt_path: str):
    import textwrap

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    emojis = ["⭐️", "🚀"]
    box_width = 50
    padding = 10
    total_width = box_width + padding * 2 + 2

    def format_line(content, emoji):
        return f"{emoji} {' ' * padding}{content.ljust(box_width)}{' ' * padding}{emoji}"

    def format_box(index, content, emoji):
        lines = textwrap.wrap(content.strip(), width=box_width)
        top = emoji * total_width
        header = format_line(f" # {index + 1} ", emoji)
        empty_line = format_line("", emoji)
        body = [format_line(line, emoji) for line in lines]
        bottom = emoji * total_width
        return "\n".join([top, header, empty_line] + body + [empty_line, bottom])

    with open(output_txt_path, "w", encoding="utf-8") as out:
        for i, key in enumerate(sorted(data, key=lambda x: int(x))):
            emoji = emojis[i % 2]
            box = format_box(i, data[key], emoji)
            out.write(box + "\n\n")



if __name__ == "__main__":
    """for testing purposes"""
    get_all_successful_jailbreaks()
    convert_json_to_readable_text("output/successful_jailbreaks_response.json", "output/successful_jailbreaks_response.txt")
