# Word2Vec

CBOW implementation of the original work by Mikotov and colleagues.

## Some Benchmarks 

- Testing the speed for naive elixir implementation of text normalization vs CLI tools, `awk` | `sed`.

```markdown
Compiling 1 file (.ex)
Operating System: macOS
CPU Information: Apple M2 Max
Number of Available Cores: 12
Available memory: 32 GB
Elixir 1.17.3
Erlang 27.1
JIT enabled: true

Benchmark suite executing with the following configuration:
warmup: 2 s
time: 5 s
memory time: 0 ns
reduction time: 0 ns
parallel: 1
inputs: none specified
Estimated total run time: 14 s

Benchmarking awk_sed ...
Benchmarking normalize_text_ex ...
Calculating statistics...
Formatting results...

Name                        ips        average  deviation         median         99th %
awk_sed                    0.58         1.74 s     ±0.19%         1.74 s         1.74 s
normalize_text_ex         0.122         8.18 s     ±0.00%         8.18 s         8.18 s

Comparison: 
awk_sed                    0.58
normalize_text_ex         0.122 - 4.71x slower +6.44 s
```

<!-- ## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `word2vecx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:word2vecx, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/word2vecx>.
 -->
