## Project Prerequisites
- Python 3.10 or 3.11
- Poetry (Follow this [Poetry installation tutorial](https://python-poetry.org/docs/#installation) to install Poetry on your system)

## Local env preparition
1. Since Python version is 3.9.6, we need to upgrade (download a high version honestly) Python version in local.
``` shell
# 使用 Homebrew 安装 pyenv 
brew install pyenv

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc # 若用 bash 则替换为 .bashrc 
echo 'eval "$(pyenv init -)"' >> ~/.zshrc 
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc

source ~/.zshrc # 或 source ~/.bashrc

# 查看可安装的 3.11 版本
pyenv install --list | grep 3.11 # 安装指定版本（例如 3.11.6） 
pyenv install 3.11.8
```

If we don't have any brew in local:
```shell
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
```

2. For Poetry, we need below pkg installed in local
   * pipx
   * Poetry: ``pipx install poetry`` 

## Installation

[project github](https://github.com/bhancockio/langchain-crash-course?tab=readme-ov-file#installation)

1. Clone the repository:
    
    ```shell
    <!-- TODO: UPDATE TO MY  -->
    git clone https://github.com/bhancockio/langchain-crash-course
    cd langchain-crash-course
    ```
    
2. Install dependencies using Poetry:
    install all dependencies configured in pyproject.toml
    ```shell
    poetry install --no-root
    ```
    
3. Set up your environment variables:
    
    - Rename the `.env.example` file to `.env` and update the variables inside with your own values. Example:
    
    ```shell
    mv .env.example .env
    ```
    
4. Activate the Poetry shell to run the examples: (like venv activate)
    
    ```shell
    poetry shell
    ```
Actually, this command has been replaced in the poetry version greater than 2.0.0
```shell
poetry env activate
source /Users/licenhao/Library/Caches/pypoetry/virtualenvs/langchain-crash-course-DbRo0LbQ-py3.11/bin/activate
```
5. Run the code examples:
    
    ```shell
     python 1_chat_models/1_chat_model_basic.py
    ```
    make sure all dependencies can be imported in the py file, maybe cannot execute correctly since we don't have openai key currently.