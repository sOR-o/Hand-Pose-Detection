import requests
import json

class SignLanguageTranslator:
    def __init__(self, default_text):
        self.default_text = default_text
        self.text = default_text
        self.queue_size = 3
        self.broken_correct_queue = []

    @staticmethod 
    def data():
        return  {
            "stream": True,
            "n_predict": 44,
            "temperature": 0.23,
            "stop": [
                "</s>",
                "Correct:",
                "Broken:"
            ],
            "repeat_last_n": 256,
            "repeat_penalty": 1.18,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "tfs_z": 1,
            "typical_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "mirostat": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
            "grammar": "",
            "n_probs": 0,
            "image_data": [],
            "cache_prompt": True,
            "api_key": "",
            "slot_id": 0,
            "prompt": ""
        }

    def append_broken_sentence(self, broken):

        if len(self.broken_correct_queue) == self.queue_size:
            self.remove_oldest_pair()
        self.broken_correct_queue.append((broken, None))
        self.update_text()

    def append_correct_sentence(self, correct):
        if not self.broken_correct_queue:
            return  
        self.broken_correct_queue[-1] = (self.broken_correct_queue[-1][0], correct)
        self.update_text()

    def remove_oldest_pair(self):
        self.broken_correct_queue.pop(0)

    def update_text(self):
        self.text = self.default_text
        for pair in self.broken_correct_queue:
            if pair[0] is not None and pair[1] is not None:
                self.text += f"\nBroken: {pair[0]}\nCorrect: {pair[1]}"
            elif pair[0] is not None:
                self.text += f"\nBroken: {pair[0]}"
            elif pair[1] is not None:
                self.text += f"\nCorrect: {pair[1]}"

    def translate(self, broken):
        self.append_broken_sentence(broken)
        nprediction = len(broken.split()) + 8

        data = self.data()
        data["prompt"] = self.text + "\nCorrect:"
        data["n_predict"] = nprediction

        response = requests.post("http://127.0.0.1:8080/completion", json=data)

        response_text = response.text

        # Extract content from each line until "\n" is encountered
        contents = []
        for line in response_text.split('\n'):
            if line.strip().startswith('data: '):
                try:
                    data = json.loads(line.split('data: ')[1])
                    content = data['content'].strip()
                    if content == "":
                        break
                    contents.append(content)
                except json.JSONDecodeError:
                    pass

        # Join the contents excluding '\n'
        extracted_text = ' '.join(contents).replace("\\n", "")
        print(extracted_text)
        self.append_correct_sentence(extracted_text)

        return extracted_text
        
        





text = "You are an AI Sign Language Translator. You are given sentences which are inferred from a vision model capturing words for signs. Expect the sentences to be broken. Your task is to transform the input sentences to the nearest correct output sentence. Even if the input sentence is grammatically incorrect, try to convey the most prominent response."
text += '''\nBroken: hi i saurabh
Correct: Hi I am Saurabh.
Broken: how are you
Correct: How are you?
Broken: where you live
Correct: Where do you live?
Broken: how much you love me
Correct: How much you love me?'''

translator = SignLanguageTranslator(text)
