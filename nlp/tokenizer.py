from typing import List,Iterable, Iterator
import regex as re
import ast 

class BPETokenizer:
    def __init__(self, vocab : dict[int,bytes], merges : list[tuple[bytes, bytes]], special_tokens : List[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary and merges and (optionally) a list of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        """
        Constructs and return a Tokenizer from a serialized vocabulary and list of merges and list of special tokens.
        """
        vocab = {}
        with open(vocab_filepath, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx_str, token_bytes = line.split('\t')
                
                vocab[ast.literal_eval(token_bytes)] = int(idx_str)

        
        merges = []
        with open(merges_filepath, "r", encoding = "utf-8" ) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                line = line.strip()

                merge_pair = re.findall(r"b'[^']*'", line)
                parts = [ast.literal_eval(t) for t in merge_pair]
                merges.append(tuple(parts))

        return cls(vocab, merges, special_tokens) 

    def encode(self, text : str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs using a robust 'Single-Pass' merge strategy.
        """
        # 1. Compile a regex that handles special tokens and normal text splits.
        special_pattern = "|".join(map(re.escape, self.special_tokens or []))
        main_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pattern = re.compile(f"({special_pattern})|({main_pattern})")

        final_byte_tokens = []

        # 2. Iterate through all the pre-tokenized chunks from the input text.
        for chunk_match in re.findall(pattern, text):
            chunk_str = next(s for s in chunk_match if s) # Find the non-empty match
            if not chunk_str:
                continue
            
            # 3. Process the chunk.
            if self.special_tokens and chunk_str in self.special_tokens:
                final_byte_tokens.append(chunk_str.encode("utf-8"))
            else:
                tokens = [bytes([b]) for b in chunk_str.encode("utf-8")]

                # 4. Sequentially apply every learned merge rule.
                for merge_pair in self.merges:
                
                    # This makes the code robust to malformed data in the merges list.
                    if not isinstance(merge_pair, tuple) or len(merge_pair) != 2:
                        continue 
                    
                    p1, p2 = merge_pair # unpacking

                    if len(tokens) < 2:
                        break
                    
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i < len(tokens) - 1 and tokens[i] == p1 and tokens[i+1] == p2:
                            new_tokens.append(p1 + p2)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens
                
                final_byte_tokens.extend(tokens)
        
        # 5. Convert the final list of byte tokens into their corresponding integer IDs.
        return [self.vocab[b] for b in final_byte_tokens if b in self.vocab]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings eg. Python file, return a generator that lazily yields tokenIDs. This is required for memory-efficient processing.
        """
        for text in iterable:
            yield self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of tokenIDs into a string.
        -> simply look up each ID's corresponding bytes and join them together [bytes form of word
        """
        id_to_byte = {v: k for k, v in self.vocab.items()}
        bytes_seq = [id_to_byte[x] for x in ids]
        words_seq = []
        for byte_seq in bytes_seq:
            word = byte_seq.decode("utf-8")
            words_seq.append(word)
        
        # print(f"id_to_byte: {id_to_byte}")
        return "".join(words_seq)

def main():
    tokenizer = BPETokenizer.from_files("vocab.txt", "merges.txt", ["<|endoftext|>"])
    test_text = ["how, are you!", "I am fine, thank you!", "Goodbye!"]
    # Sample output
    print(*tokenizer.encode_iterable(test_text))
    print(tokenizer.decode([73, 740, 2736, 44, 1436, 349, 33]))

if __name__ == "__main__":
    main()