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
  alias Req
  alias Nx
  alias Word2Vec.Utils

  @billion_words_url "https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
  @data_path "./data"
  @progress_bar_update_step 10_000

  @spec download_data(Path.t() | String.t(), String.t()) :: Req.Response.t() | nil
  def download_data(folder_path \\ @data_path, url \\ @billion_words_url) do
    save_path = folder_path |> Path.expand() |> Path.join(Path.basename(url))

    if !File.exists?(save_path) do
      Req.get!(url, into: File.stream!(save_path))
    end
  end

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
  @spec build_vocab(binary(), map()) :: map()
  def build_vocab(text, vocab \\ %{}) do
    words = String.split(text, ~r/[\n\t\s]/, trim: true)
    word_count = length(words)
    update_step = @progress_bar_update_step

    vocab =
      words
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.reduce(vocab, fn {[word0, word1], index}, acc ->
        if rem(index, update_step) == 0, do: ProgressBar.render(index, word_count, suffix: :count)

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
  @spec word_to_phrase(String.t(), map(), non_neg_integer(), non_neg_integer()) :: [String.t()]
  def word_to_phrase(text, vocab, min_count \\ 5, phrase_threshold \\ 100) do
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
        if rem(index, update_step) == 0  do
          Utils.clear_screen()
          ProgressBar.render(index, word_count, suffix: :count)
        end

        case {word0, word1} do
          {_, "\n"} ->
            "\n"

          {"\n", w1} when w1 != "\n" ->
            w1

          {w0, w1} ->
            bigram_score = score(w0, w1, vocab, min_count, word_count)
            if bigram_score > phrase_threshold, do: "_" <> w1, else: " " <> w1
        end
      end

    [Enum.at(words, 0)] ++ phrases
  end

  @spec score(binary(), binary(), map(), float(), non_neg_integer()) :: float()
  defp score(word0, word1, vocab, word_count, min_count) do
    case {Map.get(vocab, word0), Map.get(vocab, word1), Map.get(vocab, word0 <> "_" <> word1)} do
      {count0, count1, digram_count}
      when count0 > min_count and count1 > min_count and digram_count != nil ->
        (digram_count - min_count) / (count0 * count1) * word_count
      _ ->
        0.0
    end
  end

  @spec build_phrases(Path.t()) :: list()
  def build_phrases(data_path) do
    files = File.ls!(data_path)

    vocab =
      files
      |> Enum.reduce(%{}, fn file_name, acc ->
        text = File.read!(Path.join(@data_path, file_name))
        text_normalized = normalize_text(text)
        Word2Vec.build_vocab(text_normalized, acc)
      end)

    phrases =
      files
      |> Enum.map(fn file_name ->
        text = File.read!(Path.join(@data_path, file_name))
        word_to_phrase(text, vocab, 6, 50)
      end)
      |> List.flatten()

    phrases
  end
end
