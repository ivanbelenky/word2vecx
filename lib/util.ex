defmodule Word2Vec.Utils do
  # TODO: does it work?
  @spec clear_screen() :: nil
  def clear_screen() do
    IO.ANSI.clear()
    IO.ANSI.cursor(1, 1)
    nil
  end

  @spec zeros(tuple()) :: Nx.t()
  def zeros(shape) do
    Nx.broadcast(Nx.tensor(0, type: :f16), shape)
  end

  import Nx.Defn

  defn cbow_gradient(label, f, alpha) do
    (label - Nx.exp(f) / (1 + Nx.exp(f))) * alpha
  end
end
