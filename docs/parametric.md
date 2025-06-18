# Parametric grammars

- add a 64-bit parameter to symbol names
- allow rules conditional on bits being set in the parameter
- allow expressions in rules to set parameters to RHS symbols
- useful for specifying "unique", "contains" constraints in JSON schema and similar

Example - permutation of N elements:

```lark
start: list#(0)
list#(m if m & (1 << k) == 0): elt_k list#(m | (1 << k)) // repeat for k in 0..N
list#(m if m == (1 << N) - 1): ""
elt_0: ...
elt_1: ...
...
elt_N-1: ...
```

Example, any sequence of N elements, where each has to occur at least once:

```lark
start: list#(0)
list#(m if m & (1 << k) == 0): others#(m) elt_k list#(m | (1 << k)) // repeat for k in 0..N
list#(m if m == (1 << N) - 1): others#(m)
others#(m): others#(m) other#(m) | ""
others#(m if m & (1 << k) != 0): elt_k // repeat for k in 0..N
```

Expressions:

```
insert(m, k) = m | (1 << k)
remove(m, k) = m & ~(1 << k)
set(k0,...,kN) = (1 << k0) | ... | (1 << kN)
full(n) = (1 << n) - 1
```

Conditions:

```
has(m, k) = m & (1 << k) != 0
has_not(m, k) = m & (1 << k) == 0
has_all(m, n) = m == (1 << n) - 1
```

Example:

```lark
start: list#(set())
list#(m): others#(m)                           %if has_all(m, N) |
          others#(m) elt_k list#(insert(m, k)) %if has_not(m, k) // repeat for k in 0..N
others#(m): others#(m) other#(m) | ""
others#(m): elt_k %if has(m, k)  // repeat for k in 0..N
```

Nicer:

```lark
start: list{set()}
list{#m}:   others{m}                           %if has_all(m, N)
        |   others{m} elt_k list{insert(m, k)}  %if has_not(m, k) // repeat for k in 0..N
others{#m}: others{m} other{m} | ""
others{#m}: elt_k                               %if has(m, k)  // repeat for k in 0..N
```
