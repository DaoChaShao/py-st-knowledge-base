from streamlit import title, divider, expander, caption, empty, sidebar

title("Knowledge Base Chatbot")
divider()

empty_message: empty = empty()

with sidebar:
    "[![Static Badge](https://img.shields.io/badge/GitHub-py--st--deepseek--api%20in%20DaoChaShao-black?style=for-the-badge&logo=github)](https://github.com/DaoChaShao/https://github.com/DaoChaShao/py-st-knowledge-base)"

with expander("**Instructions**: Call the OpenAI model via API key.", expanded=True):
    caption("1. Clone the code from the [GitHub](https://github.com/DaoChaShao/py-st-knowledge-base) to the local.")
    caption("2. Run the command `pip install -r requirements.txt` to install the dependencies.")
    caption("3. Use the command `streamlit run main.py` to use this application.")
    caption("4. Apply for the API Key from [OpenAI](https://platform.openai.com/).")
    caption("5. Enter the API Key in the **API Key** page.")

empty_message.info("You can quick start on the **Example** page.")
