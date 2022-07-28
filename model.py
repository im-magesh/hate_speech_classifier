import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

import torch
import torch.nn as nn

from transformers import BertModel

class BertClassifier(nn.Module):
  def __init__(self,model = None,dropout=0.5,hugging_face=True):
    super(BertClassifier,self).__init__()

    if hugging_face and model is None: self.bert = BertModel.from_pretrained("bert-base-cased")
    else : self.bert = model

    self.hugging_face : bool = hugging_face

    self.linear_stack = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(768,3),
      nn.Softmax(dim=1)
    )

  def forward(self,input_id, mask):
    if self.hugging_face:
      _, pooled_output = self.bert(input_ids=input_id,attention_mask=mask,return_dict=False)
      out = self.linear_stack(pooled_output) 
      return out
    else:
      ###my mf bert
      pass
