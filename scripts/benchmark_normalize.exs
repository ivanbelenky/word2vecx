import Word2Vec, only: [normalize_text: 1, system_normalize_text: 1]

path_str = "./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"


Benchee.run(
  %{
    "normalize_text_ex" => fn ->
      text = File.read!(Path.expand(path_str))
      normalize_text(text)
    end,
    "awk_sed" => fn -> system_normalize_text(path_str) end
  }
)
