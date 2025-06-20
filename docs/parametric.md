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
other#(m if m & (1 << k) != 0): elt_k // repeat for k in 0..N
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
other#(m): elt_k %if has(m, k)  // repeat for k in 0..N
```

Nicer:

```lark
start: list{set()}
list{#m}:   others{m}                           %if has_all(m, N)
        |   others{m} elt_k list{insert(m, k)}  %if has_not(m, k) // repeat for k in 0..N
others{#m}: others{m} other{m} | ""
other{#m}: elt_k                                %if has(m, k)  // repeat for k in 0..N
```

Should also work:

```lark
start: list{set()}
list{#m}:   other{m}*                           %if has_all(m, N)
        |   other{m}* elt_k list{insert(m, k)}  %if has_not(m, k) // repeat for k in 0..N
other{#m}: elt_k                                %if has(m, k)  // repeat for k in 0..N
```

Expanded:

```lark
start: list{set()}
list{#m}:   other{m}*                        %if has_all(m, 3)
        |   other{m}* a0 list{insert(m, 0)}  %if has_not(m, 0)
        |   other{m}* a1 list{insert(m, 1)}  %if has_not(m, 1)
        |   other{m}* a2 list{insert(m, 2)}  %if has_not(m, 2)
other{#m}: a0   %if has(m, 0)
         | a1   %if has(m, 1)
         | a2   %if has(m, 2)
```

Permutation of a0,a1,a2:

```lark
start: perm{set()}
perm{#m}:   ""                        %if has_all(m, 3)
        |   a0 perm{insert(m, 0)}     %if has_not(m, 0)
        |   a1 perm{insert(m, 1)}     %if has_not(m, 1)
        |   a2 perm{insert(m, 2)}     %if has_not(m, 2)
```

## Other use cases

Let's say we want to have `s: a* b*` where `len(s) < 100`.

```lark
start  : aa::zero()
aa::n  : a aa::incr(n)    %if le(n, 100)
       | bb::n
bb::n  : b bb::incr(n)    %if le(n, 100)
       | ""
```

TODO: check on the right recursion we use above - does it cause lots of items?

## Bit range syntax

Permutation of 3 elements:

```lark
start    :  perm::0x0 "X"
perm::_  :  ""                      %if is_ones([0:3])
         |  a0 perm::set_bit(0)     %if bit_clear(0)
         |  a1 perm::set_bit(1)     %if bit_clear(1)
         |  a2 perm::set_bit(2)     %if bit_clear(2)
a0: "a"
a1: "b"
a2: "c"
```

At most 100 elements matching `a* b*`:

```lark
start  : aa::0x0
aa::_  : a aa::incr(_)    %if le(_, 100)
       | bb::n
bb::n  : b bb::incr(_)    %if le(_, 100)
       | ""
```

Unique selection of 10 elements out of N:

```lark
start    :  perm::0x0 "X"
perm::_  :  ""                      %if bit_count_ge(_, 10)
         |  a0 perm::set_bit(0)     %if and(bit_clear(0), bit_count_lt(_, 10))
         |  a1 perm::set_bit(1)     %if and(bit_clear(1), bit_count_lt(_, 10))
         |  a2 perm::set_bit(2)     %if and(bit_clear(2), bit_count_lt(_, 10))
         ...
```

At most 5 elements of each type:

```lark
start  : lst::0x0
lst::_ : a lst::incr([0:3])  %if lt([0:3], 5)
       | b lst::incr([3:6])  %if lt([3:6], 5)
       | c lst::incr([6:9])  %if lt([6:9], 5)
       | ""
```
