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
    N = {-4, -3, -2, -1, +1, +2, +3, +4}

  - skip-gram: word vector should be close to the neighbours


  """

end
