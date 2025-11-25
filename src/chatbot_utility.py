import os

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

def get_chapter_list(selected_subject):
    """
    Retrieve sorted chapter list for a given subject.
    
    Args:
        selected_subject (str): Subject name (Biology, Chemistry, or Physics)
    
    Returns:
        list: Sorted list of chapter names
    """
    subject_name = selected_subject.lower()
    chapters_dir = f"{parent_dir}/data/class_12/{subject_name}"
    
    if not os.path.exists(chapters_dir):
        print(f"Directory not found: {chapters_dir}")
        return []
    
    chapters_list = [x[:-4] for x in os.listdir(chapters_dir) if x.endswith(".pdf")]
    
    try:
        chapters_list.sort(key=lambda x: int(x.split('.')[0]))
    except (ValueError, IndexError):
        chapters_list.sort()
    
    return chapters_list


# chapters_list = get_chapter_list("Biology")
# print(chapters_list)