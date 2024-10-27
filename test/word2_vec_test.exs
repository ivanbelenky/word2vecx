defmodule Word2VecTest do
  use ExUnit.Case
  doctest Word2Vec

  test "greets the world" do
    assert Word2Vec.hello() == :world
  end
end
