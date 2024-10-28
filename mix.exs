defmodule Word2Vec.MixProject do
  use Mix.Project

  def project do
    [
      app: :word2vecx,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.6"},
      {:req, "~> 0.5.0"},
      {:benchee, "~> 1.0", only: :dev}
    ]
  end
end
