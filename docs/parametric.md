# Parametric grammars

In llguidance [grammar rules](./syntax.md) can be parameterized by a 64-bit integer.
This allows for expressing concepts like "permutation of N elements", "unique selection of N elements",
and other combinatorial structures in a concise way.

The parametrized grammars are technically still context-free, just very large:
each parametrized rule is treated as if it was expanded for each 64-bit integer value.
Of course, the grammar is only materialized lazily, during Earley parsing.

For example, this grammar describes permutations of 3 elements `a`, `b`, and `c`:

```lark
start    :  perm::0x0
perm::_  :  ""                       %if is_ones([0:3])
         |  "a" perm::set_bit(0)     %if bit_clear(0)
         |  "b" perm::set_bit(1)     %if bit_clear(1)
         |  "c" perm::set_bit(2)     %if bit_clear(2)
```

The `start` rule starts with an empty set of bits (`0x0`), and the `perm` rule expands to either an empty string
(if all bits are set, i.e., all elements have been seen)
or a choice of one of the remaining elements followed by a recursive call to `perm`
with the corresponding bit set in the parameter (using `set_bit(k)`).

Think of the `perm::_ : ...` syntax as:

```lark
perm(p)  :  ""                       %if p[0:3] == 0b111
         |  "a" perm(p | (1 << 0))   %if p[0:1] == 0b0
         |  "b" perm(p | (1 << 1))   %if p[1:2] == 0b0
         |  "c" perm(p | (1 << 2))   %if p[2:3] == 0b0
```

Where `p[x:y]` is the bit range from `x` inclusive to `y` exclusive in the parameter `p`, that is `(p >> x) & ((1 << y) - 1)`.

Currently, there is always a single parameter for each rule, and it is always a 64-bit integer.

## Function reference

The following functions are available in rule parameters. Assume current parameter is `p`,
`v` is a 64-bit integer literal using decimal or hexadecimal notation,
`k`, `x`, and `y` are bit indices (0-based).
Additionally, `_` can be used to refer to `[0:64]`.

- `_ => p` (self-reference)
- `set_bit(k) => p | (1 << k)` sets the k-th bit in the parameter
- `clear_bit(k) => p & ~(1 << k)` clears the k-th bit in the parameter
- `bit_and(v) => p & v`
- `bit_or(v) => p | v`
- `incr([x:y]) => p[x:y] == 0b11...1 ? p : p + (1 << x)` - saturating increment of bits in the range `[x:y]`
- `decr([x:y]) => p[x:y] == 0 ? p : p - (1 << x)` - saturating decrement of bits in the range `[x:y]`

The following functions are available in rule conditions (`c` is a condition expression).
All comparisons treat intergers as unsigned.

- `true` and `true()` (always true)
- `bit_clear(k) => p[k:k+1] == 0` (checks if the k-th bit is clear)
- `bit_set(k) => p[k:k+1] == 1` (checks if the k-th bit is set)
- `is_ones([x:y]) => p[x:y] == ((1 << (y - x)) - 1)` (checks if all bits in the range `[x:y]` are set)
- `is_zeros([x:y]) => p[x:y] == 0` (checks if all bits in the range `[x:y]` are clear)
- `eq([x:y], v) => p[x:y] == v` (checks if bits in the range `[x:y]` are equal to `v`)
- `ne([x:y], v) => p[x:y] != v`
- `lt([x:y], v) => p[x:y] < v`
- `le([x:y], v) => p[x:y] <= v`
- `gt([x:y], v) => p[x:y] > v`
- `ge([x:y], v) => p[x:y] >= v`
- `bit_count_eq([x:y], k) => bin(p[x:y]).count('1') == k` (checks if the number of set bits in the range `[x:y]` is equal to `k`)
- `bit_count_ne([x:y], k) => bin(p[x:y]).count('1') != k`
- `bit_count_lt([x:y], k) => bin(p[x:y]).count('1') < k`
- `bit_count_le([x:y], k) => bin(p[x:y]).count('1') <= k`
- `bit_count_gt([x:y], k) => bin(p[x:y]).count('1') > k`
- `bit_count_ge([x:y], k) => bin(p[x:y]).count('1') >= k`
- `and(c, c)` (logical AND of two conditions)
- `or(c, c)` (logical OR of two conditions)
- `not(c)` (logical negation of a condition)

## Examples

Any sequence of `a`, `b`, and `c` where each element occurs at least once:

```lark
start    :  perm::0x0
perm::_  :  ""                       %if is_ones([0:3])
         |  "a" perm::set_bit(0)
         |  "b" perm::set_bit(1)
         |  "c" perm::set_bit(2)
```

A sequence `s` matching `/a*b*/` where `len(s) < 20`:

```lark
start  : aa::0
aa::_  : "b" aa::incr(_)    %if lt(_, 20)
       | bb::_
bb::_  : "a" bb::incr(_)    %if lt(_, 20)
       | ""
```

A sequence of `a`, `b`, and `c` in any order,
where `a` and `b` can occur at most 5 times each, and `c` at most 6 times.
Note that you have to allocate enough bits for each element.

```lark
start  : lst::0x0
lst::_ : "a" lst::incr([0:3])  %if lt([0:3], 5)
       | "b" lst::incr([3:6])  %if lt([3:6], 5)
       | "c" lst::incr([6:9])  %if lt([6:9], 6)
       | ""
```

Pick at last 1 and at most 3 elements from `a`, `b`, `c`, `d`, `e`;
each element can occur at most once.

```lark
start    :  perm::0x0
perm::_  :  ""                       %if bit_count_ge(_, 1)
         |  "a" perm::set_bit(0)     %if and(bit_clear(0), bit_count_lt(_, 3))
         |  "b" perm::set_bit(1)     %if and(bit_clear(1), bit_count_lt(_, 3))
         |  "c" perm::set_bit(2)     %if and(bit_clear(2), bit_count_lt(_, 3))
         |  "d" perm::set_bit(3)     %if and(bit_clear(3), bit_count_lt(_, 3))
         |  "e" perm::set_bit(4)     %if and(bit_clear(4), bit_count_lt(_, 3))
```

## Performance considerations

All the rules above are right-recursive, which is generally [not ideal](./syntax.md#recursive-rules) for Earley parsing.
The problem is, for a list of length `N` it will generate `O(N^2)` items during parsing
(for item number `i`, it will generate about `i` items).

However, if you were to make them left-recursive, it may generate `O(2^K)` items
where `K` is the number of bits used, so do not do that.

Practically, this means the rules will not work for lists longer than about 2000 elements.
