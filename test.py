from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from jetmoe import JetMoEForCausalLM, JetMoEConfig, JetMoEForSequenceClassification
import torch

# import urllib3, socket
# from urllib3.connection import HTTPConnection
# 
# HTTPConnection.default_socket_options = (
#     HTTPConnection.default_socket_options + [
#     (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000),
#     (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
#     ])

AutoConfig.register("jetmoe", JetMoEConfig)
AutoModelForCausalLM.register(JetMoEConfig, JetMoEForCausalLM)
AutoModelForSequenceClassification.register(JetMoEConfig, JetMoEForSequenceClassification)

tokenizer = AutoTokenizer.from_pretrained('jetmoe/jetmoe-8b')
model = AutoModelForCausalLM.from_pretrained('jetmoe/jetmoe-8b', torch_dtype=torch.bfloat16)

text = "Once upon a time"
input_ids = tokenizer.encode(text, return_tensors="pt")

output = model.generate(input_ids, max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
