# Install pyenv if not installed
curl https://pyenv.run | bash

# Add pyenv to your shell (bash/zsh)
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.12.9
pyenv install 3.12.9

# Tell Poetry to use it
poetry env use $(pyenv which python3.12)
