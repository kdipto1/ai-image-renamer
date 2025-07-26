# Enhanced AI Image Renamer

This application helps you automatically rename your image files based on their content using Artificial Intelligence.

## Getting Started

Follow these steps to set up and run the application on your computer.

### Prerequisites

*   **Python:** You need Python installed on your computer. You can download it from the [official Python website](https://www.python.org/downloads/). During installation, make sure to check the box that says "Add Python to PATH".

### Setup and Installation

1.  **Download the Project:**
    *   Click on the "Code" button on the GitHub page.
    *   Select "Download ZIP".
    *   Extract the downloaded ZIP file to a folder on your computer (e.g., on your Desktop).

2.  **Open the Project Folder:**
    *   Go into the extracted folder. It will be named `Enhanced-AI-Image-Renamer-main`.

3.  **Run the Setup Script:**
    *   **For Windows:** Double-click the `setup.bat` file. This will install all the necessary components for the application.
    *   **For macOS/Linux:** Open a terminal, navigate to the project folder, and run the command: `sh setup.sh`.

    This step might take a few minutes as it downloads the required AI models and libraries.

### How to Run the Application

1.  **Start the Application:**
    *   **For Windows:** Double-click the `run.bat` file.
    *   **For macOS/Linux:** Open a terminal, navigate to the project folder, and run the command: `sh run.sh`.

2.  **Using the Renamer:**
    *   **Load the AI Model:**
        *   Choose an AI model (BLIP-2 is more accurate, BLIP-1 is faster).
        *   Click the "Load Model" button. This may take a moment, especially on the first run.
    *   **Select Your Image Folder:**
        *   Click the "Browse" button and choose the folder containing the images you want to rename.
    *   **Start Renaming:**
        *   Click the "Start Renaming" button.
    *   **Review and Confirm:**
        *   A new window will appear showing the original and new proposed filenames.
        *   Review the changes.
        *   Click "Confirm" to rename the files, or "Cancel" to go back.
    *   **Check the Log:**
        *   A `renaming_log.txt` file will be created in your selected folder, containing a record of all the file changes. You can also view the log in the "Log" tab of the application.

That's it! Your images will be renamed with descriptive names, making them easier to find and organize.
