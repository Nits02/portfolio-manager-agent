"""
Project Structure Setup Script for Portfolio Manager Agent

This script creates the necessary folder structure and placeholder files
for a Databricks multi-agent finance project.
"""

import os
import pathlib


def create_directory_with_placeholders(path: str, readme_content: str = None) -> None:
    """Create a directory with .keep file and README.md"""
    # Create directory if it doesn't exist
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # Create .keep file
    keep_file = os.path.join(path, '.keep')
    pathlib.Path(keep_file).touch()

    # Create README.md with content
    if readme_content is None:
        readme_content = f"# {os.path.basename(path)}\n\nAdd documentation for this directory here."

    with open(os.path.join(path, 'README.md'), 'w') as f:
        f.write(readme_content)


def setup_project_structure():
    """Create the main project structure with all necessary directories"""
    # Define project directories
    directories = {
        'src/agents': 'Directory for all agent-related modules and components.',
        'src/utils': 'Common utilities and helper functions.',
        'notebooks': 'Databricks notebooks for development and experimentation.',
        'streamlit_app': 'Streamlit web application code.',
        'tests': 'Test files and test utilities.',
        'docs': 'Project documentation and guides.',
        'infra': 'Infrastructure and deployment configuration.'
    }
    # Get the project root directory (where this script is located)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create each directory with its placeholders
    for dir_path, description in directories.items():
        full_path = os.path.join(project_root, dir_path)
        readme_content = (f"# {dir_path}\\n\\n{description}\\n\\n## Overview\\n\\n"
                          f"Add detailed documentation here.")
        create_directory_with_placeholders(full_path, readme_content)
        print(f"Created directory structure for: {dir_path}")


if __name__ == "__main__":
    setup_project_structure()
    print("\nProject structure setup completed successfully!")
