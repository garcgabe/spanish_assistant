import requests
from config import DEEPL_ACCESS_KEY, DEEPL_API_URL

class TranslationService:
    def __init__(self):
        self.api_key = DEEPL_ACCESS_KEY

    def translate(self, text: str, source_lang="ES", target_lang="EN") -> str:
        response = requests.post(
            url=DEEPL_API_URL,
            headers={'Authorization': f'DeepL-Auth-Key {self.api_key}'},
            data={"text": [text], "source_lang": source_lang, "target_lang": target_lang},
        )
        if response.ok:
            return response.json()["translations"][0]["text"]
        else:
            print("Translation failed:", response.text)
            return ""
