import sentencepiece as spm

tokenizer: spm.SentencePieceProcessor = spm.SentencePieceProcessor(model_file='./data/tok512.model')
