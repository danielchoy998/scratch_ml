corpus = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

def vocabulary_initialization(vocab) -> dict[str, int]:
    for byte_values in range(256):
        byte_char = chr(byte_values)
        vocab[f"{byte_char}"] = byte_values
    vocab["<|endoftext|>"] = 256
   
def pre_tokenize(words) -> list[str]:
   word_list = words.split()
   return(word_list)

def freq_table(word_list) -> dict[tuple[bytes],int]:
    freq = {}

    for word in word_list:
        bytes_tuple = tuple(c.encode("utf-8") for c in word)
        # bytes_value = tuple(bytes_string) # integer 
        freq[bytes_tuple] = freq.get(bytes_tuple, 0) + 1
    
    return freq

def merges_pair(freq_table) -> dict:
    pairs = {}
    for key, value in freq_table.items():
       i = 0
       while i < (len(key)-1):
          a,b = key[i], key[i+1] # byte_string
          pair = a + b
          pairs[pair] = pairs.get(pair,0) + value
          i+=1
    return pairs # Example : {'ab': 6, 'bc': 3, 'bd': 3, 'bCD': 2, 'CDe': 5}`
   
def top_freq_pair(pairs) -> tuple[bytes,int] : # list is becoz, there could be same frequency for different pairs, let see whether it can be improved
   if not pairs:
      raise "no pairs"
   
   max_freq = max(pairs.values())
   candidates = [k for k, v in pairs.items() if v == max_freq]

   return max(candidates)

def update_vocab(vocab, pair):
    vocab[pair] = list(vocab.values())[-1]+1
   
def update_freq(table, pair):
    new_table = {}
    for k, v in table.items():
        new_key = []
        i = 0 
        while i < len(k):
            # when i is not the last element , we check whether pair exists
            if i < len(k)- 1 and k[i] + k[i+1] == pair: 
                new_key.append(pair) 
                i+=2
            # add up the original key
            else:
                new_key.append(k[i])
                i+=1
        tuple_key = tuple(new_key)
        new_table[tuple_key] = new_table.get(tuple_key,0) + v
    
    return new_table

def main():
    vocab = {}
    vocabulary_initialization(vocab)
    print(len(vocab))
    print(vocab)

    word_list = pre_tokenize(corpus)

    print(len(word_list))
    table = freq_table(word_list) 
    print(table)
    
    # train_bpe
    for i in range(6):
        pairs = merges_pair(table)
        print(len(pairs))
        print(pairs)

        max_pair= top_freq_pair(pairs)
        print(max_pair)

        update_vocab(vocab, max_pair)
        print(len(vocab))
        table = update_freq(table, max_pair)
    
    print(len(vocab))
    print(vocab)
    
if __name__ == "__main__":
   main()




