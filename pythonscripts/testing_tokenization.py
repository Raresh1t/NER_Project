from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "A new ransomware-as-a-service (RaaS) operation named Cicada3301 has already listed 19 victims on its extortion portal, as it quickly attacked companies worldwide.\nThe new cybercrime operation is named after the mysterious 2012-2014 online/real-world game that involved elaborate cryptographic puzzles and used the same logo for promotion on cybercrime forums.\nHowever, there's no connection between the two, and the legitimate project has issued a statement to renounce any association and condemn the ransomware"
tokens = tokenizer.tokenize(sequence)
tokenids = tokenizer.encode(sequence)
decoded = tokenizer.decode(tokenids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(f'Tokens: {tokens}')
print(f'IDs: {tokenids}')
print(f'Decoded: {decoded}')