import os


def find_project_directory():
    project_dir = os.path.abspath(os.path.dirname(__file__))

    while not os.path.exists(os.path.join(project_dir, 'requirements.txt')):
        project_dir = os.path.abspath(os.path.join(project_dir, os.pardir))

        if project_dir == '/':
            raise FileNotFoundError('Could not find project root directory.')

    return project_dir


def does_file_exist(file_path):
    return os.path.exists(file_path)
