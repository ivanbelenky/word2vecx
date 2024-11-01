import Torchx
import Word2Vec
alias Nx
alias Word2Vec.Dataset

path =
  "../data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00090-of-00100"

txt = normalize_text(File.read!(path))
vocab = reduce_vocab(build_vocab(txt), 10_000)
k_dim = length(vocab)
embedding_size = 100
key = Nx.Random.key(42)
Nx.default_backend(Torchx.Backend)

{{syn0, _}, {syn1neg, _}} =
  {Nx.Random.normal(key, shape: {k_dim, embedding_size}, type: :f16),
   Nx.Random.normal(key, shape: {k_dim, embedding_size}, type: :f16)}

sentences = Dataset.build_sentences(txt, 1_000)
sentences_n = length(sentences)

vocab_list = Enum.to_list(vocab)
window = 5

sentences =
  Enum.to_list(1..sentences_n)
  |> Enum.map(fn i ->
    sentence = Enum.at(sentences, i - 1)
    Dataset.word_ctx(Enum.into(vocab_list, %{}), window, sentence)
  end)
