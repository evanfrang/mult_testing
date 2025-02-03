import re

# Input Markdown file
with open("README.md", "r", encoding="utf-8") as file:
    content = file.read()

# Regex pattern for code blocks
code_block_pattern = r"```(\w+)\n(.*?)```"

# Wrap each code block with <details> and <summary>
wrapped_content = re.sub(
    code_block_pattern,
    r"fail",
    content,
    flags=re.DOTALL,
)

# Output to a new Markdown file
with open("README_2.md", "w", encoding="utf-8") as file:
    file.write(wrapped_content)
