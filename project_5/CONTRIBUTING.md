# Contibuting to the NLP clinical text mining

## 1. Set up your environment
After forking the repository on your github,
```bash
# Clone and enter the repo
git clone https://github.com/your-username/Eneza-Data-Science-Residential-training-2026
cd Eneza-Data-Science-Residential-training-2026/project_5

# Set up a remote repository - Where the remote fetching will be occuring.
git remote main https://github.com/mkongamawe/Eneza-Data-Science-Residential-training-2026/

# Create a virtual environment (one-time)
python -m venv .venv

# Activate it (you must do this every time you open a new terminal)
c        # Linux/Mac
```

### Installing required packages
Do this only once at the start of the project. Install all the packages in the requirements.txt file

```bash
pip install -r requirements.txt
```


## 2. Working on your part
### Before every new feature

```bash
# Sync your fork with the latest changes from upstream
git checkout main
git fetch upstream
git rebase upstream/main
git push origin main
```

The first thing you need to do is to create a branch. A branch is created once for each new part of the project you are working on

```bash
# 1. Create a branch
git checkout -b feature/nltk-training
```

The next thing is to add a jupyter notebook file in the src folder of the project. Kindly ensure that the name is informative.

After finishing your work, at the end of the day, please commit the changes and push.

```bash
# 2. Commit and push to YOUR fork
git add .
git commit -m "Trained an NLTK model"
git push origin nltk-training
```

Finally, on github, open a pull request.

## 3. Project assignments
## Day 1 project assignments
| Name | Section | Description |
|------|---------|-------------|
| Clement | `Data Cleaning and filtering`| Carry out initial data cleaning on the dataset |
| Simon | `Data exploration` | Look for publicly available data and best classification algorithm |
| Godwin | `Data exploration` | Look for publicly available data and best classification algorithm |
| George | `Data exploration` | Look for publicly available data and best classification algorithm |