Lets update this with guidelines all the way from installing python and dependencies (maybe with a couple systems - anaconda (spyder, jupiter), python launcher, etc) to how to use the software and gui. We should add descriptions of every model type we have included as well. Focused descriptions on any terminology should be here to. 

# 1. Install Python
Before you begin, ensure you have Python installed. You can download it from the official Python website:
- Go to [Python Downloads](https://www.python.org/downloads/)
- Select 3.12 or less for your operating system. This will enable tensorflow 2.17.0 to work with the environment.
- Run the installer and **ensure the "Add Python to PATH" checkbox** is checked.
- In your terminal, type "alias python=python3"
  
## **If you already have Python downloaded, ensure it is 3.12 or less.** 
<br><br><br>
# 2. Install an IDE (PyCharm or Spyder, any should work though)
   
## PyCharm
- Visit the [PyCharm Download Page](https://www.jetbrains.com/pycharm/download/)
- Choose the **Community Edition** (free) or **Professional Edition** (paid).
- Download and install based on your operating system.
  
## Spyder
- Install Spyder via Anaconda (recommended for managing dependencies):
  - Install Anaconda from Anaconda's website.
  - Spyder comes pre-installed with Anaconda, but if not, you can install it using the command *conda install spyder* in the Anaconda Prompt.
# 3. Clone the GitHub Repository

**Option 1: GitHub Desktop (Easy)**

- Download and install [GitHub Desktop](https://desktop.github.com/).
- Sign in to your GitHub account.
- Find your repository, click "Clone," and choose your local path.
  
**Option 2: Git Command Line (Advanced)**

- Open your terminal (Command Prompt, PowerShell, or any terminal emulator).
- Clone the repository using the following command:

*git clone https://github.com/nanoassemblylab/saxs.git*

# 4. Create a Virtual Environment (Recommended)
Using a virtual environment ensures that your project dependencies are isolated from other projects.

In your terminal (inside the project directory), create a virtual environment:

*python -m venv venv*

This will create a folder named *venv* containing a clean Python environment.

**Activate the virtual environment:**

**Windows:**

*venv\Scripts\activate*

**Mac/Linux:**

*source venv/bin/activate*

# 5. Install Project Dependencies
Install dependencies using *pip*:
*pip install -r requirements.txt*

Alternatively, you can manually install them one by one with:

*pip install numpy*

*pip install scikit-learn*

*pip install scipy*

*pip install tensorflow*

*pip install matplotlib*



# 6. Configure the IDE
**PyCharm**

- Open PyCharm and select **Open** to load the project directory.
- In PyCharm, go to **File > Settings > Project: [Your Project] > Python Interpreter**.
- Click the gear icon and select **Add**.
- Choose **Existing environment**, and point to the *venv* folder you created earlier (usually *venv/bin/python* or *venv/Scripts/python.exe*).

  
**Spyder**

- Open Spyder (if installed via Anaconda, it should launch automatically).
- In Spyder, go to **Tools > Preferences > Python Interpreter**.
- Select the option **Use the following Python interpreter** and browse to the *python.exe* file in your *venv* directory.
- Restart Spyder to apply the changes.

  
# 7. Run the Project
Once the dependencies are installed and the IDE is set up:

- Open your project in the IDE.
- Ensure the **virtual environment** is activated.
- You can now run the project by executing the main Python script.
  - In PyCharm: Right-click the file > **Run 'main.py'**.
  - In Spyder: Click the **Run** button.
