import os

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

def get_chapter_list(selected_subject):

    if selected_subject == "Biology":
        subject_name = selected_subject.lower()
        chapters_dir = f"{parent_dir}/data/class_12/{subject_name}"
        chapters_list = os.listdir(chapters_dir)
        chapters_list = [x[:-4] for x in chapters_list]
        chapters_list.sort(key=lambda x: int(x.split('.')[0]))
        return chapters_list
    
    elif selected_subject == "Chemistry":
        subject_name = selected_subject.lower()
        chapters_dir = f"{parent_dir}/data/class_12/{subject_name}"
        chapters_list = os.listdir(chapters_dir)
        chapters_list = [x[:-4] for x in chapters_list]
        chapters_list.sort(key=lambda x: int(x.split('.')[0]))
        return chapters_list
    
    elif selected_subject == "Physics":
        subject_name = selected_subject.lower()
        chapters_dir = f"{parent_dir}/data/class_12/{subject_name}"
        chapters_list = os.listdir(chapters_dir)
        chapters_list = [x[:-4] for x in chapters_list]
        chapters_list.sort(key=lambda x: int(x.split('.')[0]))
        return chapters_list
    
    else:
        print("Select One Subject Out of Biology, Chemistry, Physics")


# chapters_list = get_chapter_list("Biology")
# print(chapters_list)