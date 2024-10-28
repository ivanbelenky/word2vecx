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

  @billion_words_url "https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
  @data_path "./data"

  @spec download_data(
    save_path :: Path.t() | String.t(),
    url :: String.t()
  ) :: Req.Response.t() | nil
  def download_data(folder_path \\ @data_path, url \\ @billion_words_url) do
    save_path = folder_path |> Path.expand |> Path.join(Path.basename(url))
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

  @spec system_normalize_text(path :: Path.t()) :: String.t()
  def system_normalize_text(path) do
    System.cmd("bash", ["./scripts/normalize_mikotov.sh", path])
    |> elem(0)
  end


end
