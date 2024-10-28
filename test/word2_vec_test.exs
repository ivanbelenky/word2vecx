defmodule Word2VecTest do
  use ExUnit.Case
  doctest Word2Vec

  @example_word_file "./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"

  test "normalizers are equal" do
    fp = Path.expand(@example_word_file)
    txt = File.read!(fp)
    system_normalized = Word2Vec.system_normalize_text(Path.expand(@example_word_file))
    elixir_normalized = Word2Vec.normalize_text(txt)
    system_normalized = String.replace(system_normalized, ~r/\s+/, " ")
    elixir_normalized = String.replace(elixir_normalized, ~r/\s+/, " ")
    assert system_normalized == elixir_normalized
  end
end
