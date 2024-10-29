defmodule Word2Vec.Utils do
  # TODO: does it work?
  @spec clear_screen() :: nil
  def clear_screen() do
    IO.ANSI.clear()
    IO.ANSI.cursor(1, 1)
    nil
  end
end
