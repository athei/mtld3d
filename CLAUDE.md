# Working on mtld3d

[`CONTRIBUTING.md`](CONTRIBUTING.md) is the operating manual for this repository:
the gates, how to read their output, how conformance work is organised, and what
a pull request is expected to contain. Read it before changing anything.

It points at the files that own the rules:

- [`README.md`](README.md), the goal, the requirements, and how to build,
  install, configure and log.
- [`docs/STATUS.md`](docs/STATUS.md), what is implemented, what is not, and
  the divergences kept on purpose.
- [`CONTRIBUTING.md`](CONTRIBUTING.md), the workflow and the lessons that no
  other file owns.
- [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md), every code rule, with
  `make audit` enforcing the mechanical half.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), the PE/Unix boundary contract,
  the threading model, and the diagnostic toolkits.
- [`unix/conformance/CONFORMANCE.md`](unix/conformance/CONFORMANCE.md), the
  conformance suite, the classification scheme, and the rationale behind every
  divergence we keep.
- [`windows/tests/COVERAGE.md`](windows/tests/COVERAGE.md), what the end-to-end
  suite covers and which stubs it pins.
- [`mtld3d.conf`](mtld3d.conf), every runtime option with its default.

`make check` and `make test` are the gates. `CONTRIBUTING.md` explains how to
read a test run, which matters here: the runner is fail-fast and its summary
undercounts.
