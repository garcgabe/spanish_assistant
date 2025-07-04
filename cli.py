from services.translator import TranslationService
from services.ai_service import AIService
from services.audio_recorder import record_audio_cli
from prompts import CONVO_PROMPT
import os



def main():
    translator = TranslationService()
    ai = AIService()
    context = [{"role": "system", "content": CONVO_PROMPT}]

    print("Choose input mode: [1] Text [2] Voice")
    choice = input("> ")

    while True:
        if choice == '1':
            text = input("Tu mensaje en español: ")
        else:
            audio_path = record_audio_cli()
            text = ai.transcribe_audio(audio_path)
            print("Transcribed:", text)

        if text.lower() in ['quit', 'exit']:
            break

        translation = translator.translate(text)
        print("English:", translation)

        if choice != '1':
            context.append({"role": "user", "content": text})
            response = ai.get_text_completion(context)
            print("AI:", response)

            speech = ai.text_to_speech(response)
            ai.play_audio(speech)
            os.remove(speech)

            context.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
