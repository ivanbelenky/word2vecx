defmodule Word2Vec do
  @moduledoc """
  `Word2Vec` implementation in elixir.

  ## Mathematical details

  The idea of word2vec as the name suggests is to build a representation
  for each word individually in a vector space of `n` dimensions. Therefore
  one says that there is a vocabulary V, belonging to a corpus C. And
  our goal is to learn a vector `v_w` in R^n for all w in V.

  There are two main mathematical models to apply.
  - CBOW: sum of neighbors should be close to the vector of the word
    N = {-4, -3, -2, -1, +1, +2, +3, +4} is the index set of words that
    will be considered the neighborhood of word at index 0. The training
    objective is to maximize the quantity:

    prod_{i in C}( Pr(w_i | w_j: j in N+i) --> stabilize via log
    sum_{i in C}(ln Pr(w_i | w_j: j in N+i) --> where probability is
    Pr(w | w_j: j in N + i) := softmax(v*v_w) where
    v = sum_j(v_{w_j}) and
    v_w = {v(w): w in V}

    So essentially is calculating a probability for every single
    word in the vocabulary, there will be a probability. After
    applying a logarithm, you end up with

    ln Pr(w | wj: j in N+i) = <v_w, v_wj> - ln sum_w( exp(<v_w, v_wj>) )

    therefore we can specify the optimization problem as the maximization
    of the sum over all the corpus of the above probability. The
    compounded, and simplectic addition of probabilities results in
    the desired output.

    // TODO: what about punctuation symbols and such? How does
    // the corpus look like int this regard

  - skip-gram: word vector should be close to the neighbours
    with the same set N as the CBOW model the idea behind the skip gram model
    is to maximize the probability of close words given the 0th index vocabulary.
    This is

    prod_{i in C} = prod( Pr(w_j: j in N+i | w_i) )
    we transform to a logarithm for numerical stability, but the essence is exactly
    the same. Maximizing this optimization function, gives rise to pretty much the same
    function with a slight difference

    ln Pr(w | wj: j in N+i) = <v_w, v_wj> - ln sum_w( exp(<v_w, v_wi>) )

  There is a very nice [paper](https://arxiv.org/pdf/1402.3722) explaining in clearer
  terms the ideas behind Mikolov and colleagues paper, negative sampling, and the overall
  optimization algorithm that needs to be used.
  """
  alias Word2Vec.Utils
  alias Word2Vec.Dataset

  @progress_bar_update_step 10_000

  @spec normalize_text(text :: String.t()) :: String.t()
  def normalize_text(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[’′]/u, "'")
    |> String.replace(~r/''/u, " ")
    |> String.replace(~r/'/u, " ' ")
    |> String.replace(~r/[“”]/u, "\"")
    |> String.replace(~r/"/u, " \" ")
    |> String.replace(~r/\./u, " . ")
    |> String.replace(~r/<br \/>/u, " ")
    |> String.replace(~r/,/u, " , ")
    |> String.replace(~r/\(/u, " ( ")
    |> String.replace(~r/\)/u, " ) ")
    |> String.replace(~r/!/u, " ! ")
    |> String.replace(~r/\?/u, " ? ")
    |> String.replace(~r/;/u, " ")
    |> String.replace(~r/:/u, " ")
    |> String.replace(~r/-/u, " - ")
    |> String.replace(~r/=/u, " ")
    |> String.replace(~r/\*/u, " ")
    |> String.replace(~r/\|/u, " ")
    |> String.replace(~r/«/u, " ")
    |> String.replace(~r/[0-9]/u, " ")
  end

  @spec system_normalize_text(Path.t()) :: String.t()
  def system_normalize_text(path) do
    System.cmd("bash", ["./scripts/normalize_mikotov.sh", path])
    |> elem(0)
  end

  @doc """
  A vocabulary is nothing more than a map between words and its
  count. Rigorously the keys of the map form the vocabulary, and the
  map is the counter of the vocabulary. This function is somewhat slow
  for big files. Probably there is room for optimizing.
  """
  @spec build_vocab(binary(), map(), boolean()) :: map()
  def build_vocab(text, vocab \\ %{}, verbose \\ false) do
    words = String.split(text, ~r/[\n\t\s]/, trim: true)
    word_count = length(words)
    update_step = @progress_bar_update_step

    vocab =
      words
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.reduce(vocab, fn {[word0, word1], index}, acc ->
        if rem(index + 1, update_step) == 0 and verbose,
          do: ProgressBar.render(index, word_count, suffix: :count)

        acc
        |> Map.update(word0, 1, &(&1 + 1))
        |> Map.update(word0 <> "_" <> word1, 1, &(&1 + 1))
      end)

    vocab
  end

  @doc """
  The idea of word2phrase is to generate a new formatted text where
  words that should go together end up together. Apparently this is a nice
  legacy way of building up and creating phrases based on digram probabilities
  """
  @spec word_to_phrase(String.t(), map(), non_neg_integer(), non_neg_integer(), boolean()) :: [
          String.t()
        ]
  def word_to_phrase(text, vocab, min_count \\ 5, phrase_threshold \\ 100, verbose \\ false) do
    # The idea is  to traverse this words, by chunks of 2. That is
    # [word_i, word1_i+1].
    # On each iteration we need to decide if we should write either of
    # - " " <> word_i+1 or
    # - "_" <> word_i+1
    # this depends on the score of the bigram word_i <> "_" <> word_i+1
    # if word_i+1 is \n --> we add \n
    # if word_i is \n --> we write word_i+1 unless is another \n
    words = String.split(text, ~r/[\t\s]/, trim: true)
    # Calculate length only once
    word_count = length(words)
    update_step = @progress_bar_update_step

    phrases =
      for {[word0, word1], index} <- Enum.with_index(Enum.chunk_every(words, 2, 1, :discard)) do
        if rem(index + 1, update_step) == 0 and verbose do
          ProgressBar.render(index, word_count, suffix: :count)
        end

        case {word0, word1} do
          {_, "\n"} ->
            "\n"

          {"\n", w1} when w1 != "\n" ->
            w1

          {w0, w1} ->
            bigram_score = score(w0, w1, vocab, min_count, word_count)
            if bigram_score > phrase_threshold do
              "_" <> w1
            else
               " " <> w1
            end
        end
      end

    [Enum.at(words, 0)] ++ phrases
  end

  @spec score(binary(), binary(), map(), float(), non_neg_integer()) :: float()
  defp score(word0, word1, vocab, min_count, word_count) do
    case {Map.get(vocab, word0), Map.get(vocab, word1), Map.get(vocab, word0 <> "_" <> word1)} do
      {count0, count1, digram_count}
      when count0 != nil and count0 > min_count and count1 != nil and count1 > min_count and digram_count != nil ->
        (digram_count - min_count) / (count0 * count1) * word_count
      _ ->
        0.0
    end
  end

  @spec build_vocab_file(Path.t(), non_neg_integer()) :: map()
  def build_vocab_file(file_path, threads_n \\ 100, subchunks \\ 500) do
    file_size = File.stat!(file_path).size
    chunk_size = div(file_size, threads_n)

    vocab =
      1..threads_n
      |> Task.async_stream(
        fn i ->
          subchunk_size = div(chunk_size, subchunks)
          file = File.open!(file_path, [:read])

          vocab =
            Enum.reduce(0..subchunks, %{}, fn j, acc ->
              start_pos = chunk_size * (i - 1) + subchunk_size * j
              :file.position(file, start_pos)

              content =
                case IO.read(file, subchunk_size) do
                  :eof -> ""
                  txt -> txt
                end

              until_line =
                case IO.read(file, :line) do
                  :eof -> ""
                  txt -> txt
                end

              text = content <> until_line
              build_vocab(text, acc)
            end)
          vocab
        end,
        timeout: 3_600_000
      )
      |> Enum.reduce(%{}, fn {:ok, v}, acc -> merge_vocabs(acc, v) end)

    vocab
  end

  @spec build_phrases_file(Path.t(), non_neg_integer()) :: list(binary())
  def build_phrases_file(
        file_path,
        vocab,
        min_count \\ 5,
        phrase_threshold \\ 100,
        threads_n \\ 16,
        subchunks \\ 64
      ) do
    file_size = File.stat!(file_path).size
    chunk_size = div(file_size, threads_n)

    phrases =
      1..threads_n
      |> Task.async_stream(
        fn i ->
          subchunk_size = div(chunk_size, subchunks)
          file = File.open!(file_path, [:read])

          phrases =
            Enum.reduce(0..subchunks, [], fn j, acc ->
              start_pos = chunk_size * (i - 1) + subchunk_size * j
              :file.position(file, start_pos)

              content =
                case IO.read(file, subchunk_size) do
                  :eof -> ""
                  txt -> txt
                end

              until_line =
                case IO.read(file, :line) do
                  :eof -> ""
                  txt -> txt
                end

              text = content <> until_line
              acc ++ word_to_phrase(text, vocab, min_count, phrase_threshold)
            end)

          IO.puts("thread #{i} done!\n")
          phrases
        end,
        timeout: 3_600_000
      )
      |> Enum.reduce([], fn {:ok, p}, acc -> acc ++ p end)

    phrases
  end

  def merge_vocabs(vocab1, vocab2) do
    Map.merge(vocab1, vocab2, fn _key, value1, value2 ->
      value1 + value2
    end)
  end

  def create_train_files(file_path) do
    # expecting 5GB file approximately
    file_content = File.read!(file_path)
    vocab = build_vocab(file_content, %{}, true)

    phrases = word_to_phrase(file_content, vocab, 10, 200, true)
    data_phrase_file = File.open!(Path.join(@data_path, "data_phrases"), [:write, :utf8])
    IO.write(data_phrase_file, Enum.join(phrases, ""))
    :file.position(data_phrase_file, 0)

    data_phrases_content = IO.read(data_phrase_file, :eof)
    vocab2 = build_vocab(data_phrases_content, %{}, true)
    phrases2 = word_to_phrase(data_phrases_content, vocab2, 10, 100, true)

    data_phrases2_file = File.open!(Path.join(@data_path, "data_phrases2"), [:write, :utf8])
    IO.write(data_phrases2_file, Enum.join(phrases2, ""))
  end

  def reduce_vocab(vocab, max_size) do
    {vocab, _} = vocab |> Enum.sort_by(fn {_word, count} -> count end, :desc)|> Enum.split(max_size)
    vocab
  end

  def train(input_file_path, embedding_size \\ 500, max_vocab_size \\ 100_000, window \\ 5, alpha \\ 0.025) do
    data_content = File.read!(input_file_path)
    vocab = build_vocab(data_content, %{}, true)

    key = Nx.Random.key(42)
    vocab = reduce_vocab(vocab, max_vocab_size)

    k_dim = map_size(vocab)
    {{syn0, _}, {syn1neg, _}} =
      {Nx.Random.normal(key, shape: {k_dim, embedding_size}, type: :f16),
       Nx.Random.normal(key, shape: {k_dim, embedding_size}, type: :f16)}

    sentences = Dataset.build_sentences(data_content, 1_000)
    sentences_n = length(sentences)
    1..sentences_n
      |> Enum.map(fn i ->
        case Dataset.word_ctx(vocab, window) do
          :end -> nil
          {:batch, words_ctxs} ->
            for {word, ctx} <- words_ctxs do
              # direct "access" to memory for addition, Idk if this is optimal
              # but I am trying to replicate as much as possible mikotov's
              # implementation
              {neu1, _} = Utils.zeros({1, embedding_size})
              {neu1error, _} = Utils.zeros({1, embedding_size})

              # W^T * (sum_i x_i)/C
              ctx_size = length(ctx)
              neu1 = Enum.reduce(ctx, neu1, fn ctx_word_index, acc ->
                Nx.divide(ctx_size, Nx.add(acc, syn0[ctx_word_index]))
              end)

              # f is the value at softmax, it is <hidden, syn1_{i*,j}> where
              # i* is the index for word
              neg_samples = Enum.map(Dataset.negative_samples(word, vocab), &({0, &1}))

              neu1error = Enum.reduce([{word, 1} | Enum.map(neg_samples, &({0, &1}))], neu1error, fn {idx, label}, acc ->
                f = Nx.dot(neu1, Nx.reshape(syn1neg[word], {embedding_size, 1}))
                g = (label - Nx.exp(f)/(1+Nx.exp(f))) * alpha
                inner_neu1e = Nx.add(acc, Nx.multiply(g, syn1neg[idx]))
                learn_neg = Nx.add(syn1neg[idx], Nx.multiply(g, neu1))
                Nx.put_slice(syn1neg, [idx, 0], Nx.reshape(learn_neg, {1, embedding_size})) # inplace
                inner_neu1e
              end)
              Nx.put_slice(syn0, [word, 0], Nx.add(syn0[word], neu1error)) # inplace
            end
          _ -> nil
          ProgressBar.render(i, sentences_n, suffix: :count)
        end
      end)


  end

  def hidden(syn0, x) do
    Nx.dot(Nx.transpose(syn0), x)
  end

end
