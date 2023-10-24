from itertools import chain, combinations
import transformers
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch

class UtteranceEncoding():

    def __init__(self):
        # initialize possible blocks
        self.blocks = ['red block',
                        'blue block',
                        'green block',
                        'purple block',
                        'yellow block']
                        
        # initialize possible weights
        self.weights = ['ten',
                        'twenty',
                        'thirty',
                        'forty',
                        'fifty']
                        
        # initialize possible relations
        self.relations = ['equals',
                        'does not equal',
                        'is less than',
                        'is more than']
                        
        self.props = []
        self.prop_embs = {}

        # initialize possible propositions (Cartesian product of blocks, weights, and relations)
        self.generate_props()
        
        # Load pre-trained model tokenizer (vocabulary)
        self.bert_base_uncased_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load pre-trained model (weights)
        self.bert_base_uncased_model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states
                                  )
        
        for i in range(len(self.props)):
            print(f"{i} Encoding {self.props[i]}")
            tokenized_text, seq_embedding, pooled_embedding, hidden_states = \
                self.get_sentence_embedding(self.props[i], self.bert_base_uncased_tokenizer, self.bert_base_uncased_model)
            token_embeddings = torch.squeeze(torch.stack(hidden_states, dim=0), dim=1).permute(1,0,2)
            sum_embeddings = self.get_token_embeddings(token_embeddings, 4)
            self.prop_embs[self.props[i]] = torch.mean(torch.stack(sum_embeddings),axis=0)
            
        
    '''Get the embedding for a single sentence'''
    def get_sentence_embedding(self, text, tokenizer, model, verbose=False):
        
        # Add the special tokens
        marked_text = text
        if isinstance(model,BertModel):
            marked_text = "[CLS] " + text + " [SEP]"
        
        # Split the sentence into tokens
        tokenized_text = tokenizer.tokenize(marked_text)
        
        # Map the token strings to their vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        if verbose:
            # Display the words with their indices
            for i, tup in enumerate(zip(tokenized_text, indexed_tokens)):
                print('{:} {:<12} {:>6,}'.format(i, tup[0], tup[1]))
                
        # Mark each token as belonging to sentence "1"
        segments_ids = [1] * len(tokenized_text)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        model.eval()
        
        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers.
        with torch.no_grad():

            outputs = model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]
          
        if verbose:
            print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
        layer_i = 0

        if verbose:
            print("Number of batches:", len(hidden_states[layer_i]))
        batch_i = 0

        if verbose:
            print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        token_i = 0

        if verbose:
            print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
        
        return tokenized_text, outputs[0], outputs[1], hidden_states
        
        
    '''Returns embeddings for tokens and indices in "indices", concatenated and summed over the last n layers'''
    def get_token_embeddings(self, token_embeddings, last_n_layers):
        
        # Stores the token vectors
        token_vecs_sum = []
            
        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-last_n_layers:], dim=0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        print ('Summed embeddings shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
            
        return token_vecs_sum

        
    def findsubsets(self, s, n):
        return list(combinations(s, n))
        
        
    def powerset(self,iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        

    def generate_props(self):
        for block in self.blocks:
            for relation in self.relations:
                for weight in self.weights:
                    prop = f"{block} {relation} {weight}"
                    self.props.append(prop)
                    print(prop)
                block_combos = self.powerset(self.blocks[:self.blocks.index(block)]+\
                    self.blocks[self.blocks.index(block)+1:])
                for sum in [" plus ".join(c) for c in [*block_combos][1:]]:
                    prop = f"{block} {relation} {sum}"
                    self.props.append(prop)
                    print(prop)
        subsets = list(self.findsubsets([p for p in self.props if ' equals ' in p and not p.endswith('block')],3))
        for block in self.blocks:
            to_remove = []
            for i in range(len(subsets)):
                if len([item for item in subsets[i] if item.startswith(block)]) > 1:
                    to_remove.append(i)
            to_remove.reverse()
            for i in to_remove:
                subsets.pop(i)
        for subset in subsets:
            prop = " and ".join(subset)
            self.props.append(prop)
            print(prop)
        print(len(self.props))
