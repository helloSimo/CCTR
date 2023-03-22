from transformers import AutoTokenizer, PreTrainedTokenizer


tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

print(tokenizer(' [unused0] ', return_tensors='pt').input_ids)
print(tokenizer(' [unused1] ', return_tensors='pt').input_ids)
print(tokenizer(' [unused2] ', return_tensors='pt').input_ids)
print(tokenizer(' [2] ', return_tensors='pt').token_type_ids)
