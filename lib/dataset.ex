defmodule Word2Vec.Dataset do

  @billion_words_url "https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
  @data_path "./data"

  @spec download_data(Path.t() | String.t(), String.t()) :: Req.Response.t() | nil
  def download_data(folder_path \\ @data_path, url \\ @billion_words_url) do
    save_path = folder_path |> Path.expand() |> Path.join(Path.basename(url))

    if !File.exists?(save_path) do
      Req.get!(url, into: File.stream!(save_path))
    end
  end

  @spec build_sentences_rec_slow(binary(), non_neg_integer()) :: [String.t()]
  def build_sentences_rec_slow(content, max_length_sentences) do
    if String.length(content) < max_length_sentences do
      [content]
    else
      {first_sentence, rest} = String.split_at(content, max_length_sentences)
      if String.starts_with?(rest, " ") do
        [first_sentence | build_sentences_rec_slow(rest, max_length_sentences)]
      else
        [first_rest | rest] = String.split(rest, " ")
        [
          first_sentence <> first_rest |
          build_sentences_rec_slow(Enum.join(rest, " "), max_length_sentences)]
      end
    end
  end

  @spec build_sentences_slow(binary(), non_neg_integer()) :: [String.t()]
  def build_sentences_slow(text, max_length_sentences) do
    String.split(text, "\n")
      |> Enum.map(fn line ->
        case String.length(line) < max_length_sentences do
          true -> line
          false -> build_sentences_rec_slow(line, max_length_sentences)
        end
      end)
      |> List.flatten()
  end


  @spec build_sentences(binary(), non_neg_integer(), binary()) :: [String.t()]
  def build_sentences(content, max_length, acc \\ "")

  def build_sentences(<<>>, _max_length, acc), do: [acc]
  def build_sentences(<<char::utf8, rest::binary>>, max_length, acc) do
    cond do
      byte_size(acc) >= max_length ->
        case char do
          ?\s -> [acc | build_sentences(rest, max_length, "")]
          _ -> find_word_boundary(rest, max_length, acc <> <<char::utf8>>)
        end
      true ->
        build_sentences(rest, max_length, acc <> <<char::utf8>>)
    end
  end

  defp find_word_boundary(<<?\s, rest::binary>>, max_length, acc) do
    [acc | build_sentences(rest, max_length, "")]
  end
  defp find_word_boundary(<<char::utf8, rest::binary>>, max_length, acc) do
    find_word_boundary(rest, max_length, acc <> <<char::utf8>>)
  end
  defp find_word_boundary(<<>>, _max_length, acc), do: [acc]

  # Negative samples creation
  @spec negative_samples(non_neg_integer(), [tuple()]) :: [non_neg_integer()]
  def negative_samples(word_idx, vocab_list, n \\ 5) do
    Enum.take_random(0..n, n)
    |> Enum.filter(fn idx ->
      {random_word, _} = Enum.at(vocab_list, idx, "~~None~~")
      {word, _} = Enum.at(vocab_list, word_idx, "~~None~~")
      random_word != word
    end)
  end

  @spec word_ctx(map(), non_neg_integer(), binary()) :: nil
  def word_ctx(_vocab, _window, <<>>), do: :end

  def word_ctx(vocab_map, window, sentence) do
    word_list = String.split(sentence) |> Enum.filter(fn w -> vocab_map[w] != nil end)
    n = length(word_list)
    window = if window < n, do: window, else: n
    words_ctxs =
      Enum.to_list(0..n-1)
      |> Enum.map(fn i ->
        {start, fin} = {Kernel.max(0, i-window), Kernel.min(n-1, i+window)}
        {
          Enum.at(word_list, i),
          Enum.to_list(start..fin)
          |> Enum.filter(fn idx -> idx != i end)
          |> Enum.map(fn j -> Enum.at(word_list, j) end)
        }
      end)
    {:batch, words_ctxs}
  end
end
