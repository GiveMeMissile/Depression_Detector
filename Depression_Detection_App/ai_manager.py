import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import math


TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
MODEL_1 = "Depression_detecting_models/Model_001.pth"
MODEL_1_STATE_DICT = "Depression_detecting_models/Model_001_state_dict.pth"
PRETRAINED_MODEL = "bert-base-uncased"
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_FEATURES = 512
HIDDEN_LAYERS = 6
DROPOUT = .2
OUTPUT_FEATURES = 1


class ClassifierModel(nn.Module):
    def __init__(self, classes, pretrained_model):
        super(ClassifierModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, classes)

    def forward(self, x, padding_attention_mask=None):
        x = self.pretrained_model(input_ids=x, attention_mask=padding_attention_mask)
        x = self.classifier(x.pooler_output)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, dimensions, dropout, max_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_length, dimensions)
        positions_list = torch.arange(0, max_length, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dimensions, 2).float() * (-math.log(10000.0))/dimensions)

        pos_encoding[:, 0::2] = torch.sin(positions_list*division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        return self.dropout(x + self.pos_encoding[:x.size(0), :])


class TransformerModel(nn.Module):
    def __init__(self, num_tokens, hidden_features, num_heads, num_layers, dropout, output):
        super(TransformerModel, self).__init__()

        self.hidden_features = hidden_features

        self.positional_encoder = PositionalEncoding(hidden_features, dropout, max_length=MAX_LENGTH)

        self.embedding = nn.Embedding(num_tokens, hidden_features)
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_features,
            nhead=num_heads,
            dim_feedforward=HIDDEN_FEATURES,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_features, output)

    def forward(self, x, padding_attention_mask=None):
        x = x.to(DEVICE)
        x = self.embedding(x.long()) * math.sqrt(self.hidden_features)
        x = self.positional_encoder(x)

        # x = x.permute(1, 0, 2)

        try:
            x = self.transformer(x, src_key_padding_mask=padding_attention_mask)
        except Exception as e:
            print(f"attention mask: {padding_attention_mask}")
            print(f"x: {x}")
            print("shape of x: ", x.shape)
            print(f"shape of padding_attention_mask: {padding_attention_mask.shape}")
            raise e
        x = self.transformer(x, src_key_padding_mask=padding_attention_mask)
        print(x.shape)
        x = self.output_layer(x.mean(dim=1))
        return x


def initilize_models():
    
    bert = ClassifierModel(1, AutoModel.from_pretrained(PRETRAINED_MODEL))
    transformer = TransformerModel(
        len(TOKENIZER),
        HIDDEN_FEATURES,
        8,
        HIDDEN_LAYERS,
        DROPOUT,
        OUTPUT_FEATURES
    )
    return transformer, bert


# Imma finish this later trust
class DepressionDetector:

    def __init__(self):
        transformer, bert = initilize_models()
        self.tokenizer = TOKENIZER
        try:
            model_1 = torch.load(MODEL_1, weights_only=False)
        except Exception:
            print("Oops")
            model_1 = transformer
            state_dict = torch.load(MODEL_1_STATE_DICT, weights_only=False)
            model_1.load_state_dict(state_dict)

        self.model_dict = {"model_1": model_1.to(DEVICE)}

    def forward(self, model, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = self.split_and_pad_text(tokens["input_ids"].squeeze(dim=0))
        with torch.inference_mode():
            attention_mask = self.create_attention_mask(tokens)
            tokens = tokens.to(DEVICE)
            logits = self.model_dict[model](tokens, padding_attention_mask=attention_mask)
            return self.convert_to_prob(logits)

    def split_and_pad_text(self, tokens):
        batch = []
        max_alter = 0
        i = 0
        while True:
            i += 1
            if tokens.shape[0] == MAX_LENGTH:
                batch.append(tokens)
                break
            if tokens.shape[0] < MAX_LENGTH:
                zeros = torch.zeros(MAX_LENGTH - tokens.shape[0])
                tokens_batch = torch.cat([tokens, zeros])
                batch.append(tokens_batch)
                break
            else:
                tokens_batch = tokens[:MAX_LENGTH]
                batch.append(tokens_batch)
                tokens = tokens[max_alter+MAX_LENGTH:]
                max_alter += MAX_LENGTH

        tokens = torch.stack(batch, dim=0)
        return tokens
    
    def create_attention_mask(self, X):
        attention_mask = (X != 0)
        attention_mask = attention_mask.to(DEVICE)
        return attention_mask
    
    def convert_to_prob(self, logits):
        prob = torch.sigmoid(logits)
        prob = prob.sum(dim=0)/logits.size(0)
        return prob.item()
    
x = DepressionDetector()
print(x.forward("model_1", "XCGUHIJOHGFDSAZEXFCGVbgtXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdXCGUHIJOHGFDSAZEXFCGVbgtRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&FtdRDRSXFGCHVBUhiyt7r6dtrXCGVUHIytfdtrxfcGVHUHIYT&Ftd"))
    