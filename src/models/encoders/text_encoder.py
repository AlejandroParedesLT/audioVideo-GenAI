from transformers import AutoTokenizer, T5EncoderModel

class T5TextEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        self.model = T5EncoderModel.from_pretrained("google-t5/t5-small")
    
    def encode_text(self, prompt):
        input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids
        outputs = self.model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

