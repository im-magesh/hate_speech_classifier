import torch
import torch.nn

from .model import BertClassifier
from transformers import BertTokenizer, BertModel
#from .remove_logs import set_global_logging_level 

model = BertClassifier()
model.load_state_dict(torch.load("HateClassifier/weights/BERTClassifierWeights.pt"))

def inference(sentence : str, model : BertClassifier = model) -> dict:
  with torch.no_grad():
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    embedded = tokenizer(sentence,padding="max_length",max_length=512,truncation=True,return_tensors="pt")

    mask = embedded["attention_mask"]
    token_ids = embedded["input_ids"].squeeze(1)

    print(token_ids.shape)
    print(mask.shape)
    output = model(token_ids,mask)

    #convert this dict to json object if you're going to send it as response
    return {
        "input sentence":sentence,
        "vals":{
          "hatespeech":round(output[0][2].item(),2),
          "offensive":round(output[0][0].item(),2),
          "normal":round(output[0][1].item(),2)
        }
    }

