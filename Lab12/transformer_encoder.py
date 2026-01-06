import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)



def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe



def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    return np.matmul(weights, V)



class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.WQ = np.random.randn(d_model, d_model)
        self.WK = np.random.randn(d_model, d_model)
        self.WV = np.random.randn(d_model, d_model)
        self.WO = np.random.randn(d_model, d_model)

    def split_heads(self, x):
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.num_heads, self.d_k)

    def forward(self, x):
        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        heads = []
        for i in range(self.num_heads):
            head = scaled_dot_product_attention(Q[:, i], K[:, i], V[:, i])
            heads.append(head)

        concat = np.concatenate(heads, axis=-1)
        return concat @ self.WO



class FeedForward:
    def __init__(self, d_model):
        self.W1 = np.random.randn(d_model, d_model * 4)
        self.W2 = np.random.randn(d_model * 4, d_model)

    def forward(self, x):
        x = np.maximum(0, x @ self.W1)   # ReLU
        return x @ self.W2



class TransformerEncoderBlock:
    def __init__(self, d_model, num_heads):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        # Multi-head self attention
        attn_output = self.attention.forward(x)
        x = layer_norm(x + attn_output)

        # Feed forward network
        ffn_output = self.ffn.forward(x)
        x = layer_norm(x + ffn_output)

        return x



if __name__ == "__main__":
    seq_len = 6
    d_model = 8
    num_heads = 2

    # Dummy input embeddings
    embeddings = np.random.randn(seq_len, d_model)

    # Add positional encoding
    x = embeddings + positional_encoding(seq_len, d_model)

    encoder = TransformerEncoderBlock(d_model, num_heads)
    output = encoder.forward(x)

    print("Encoder Output Shape:", output.shape)
