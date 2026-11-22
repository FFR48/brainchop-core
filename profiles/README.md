# Firefox wasm CPU profile

The 125.8 s result belongs to Playwright's patched Firefox Nightly, not Firefox
generally. Installed Firefox 153.0.1 ran the same symbol-mapped model16 module
and `t1_2mm` fixture bit-exact in 7.6 s through browser startup and inference.

CLAUDE.md's *Known gaps* owns the finding and the standing instruction not to
tune production code against the patched Nightly; what is here is the capture's
own provenance and how to make another one.

The local capture `firefox-wasm-cpu-symbols.json` is intentionally ignored (41
MiB), so a clone gets these recipes rather than the recording. It sampled every
registered thread at 1 ms. Mapping its Wasm function indices with the
`.js.symbols` file produced:

| leaf frame | samples |
|---|---:|
| `mn_conv_hidden_rows` | 46,920 |
| `mn_classify_rows` | 14,813 |
| `mn_norm_rows` | 2,596 |
| `mn_moments_rows` | 597 |

All 14 pool workers were active. Their sampled CPU time was balanced at roughly
8.1–10.1 seconds each; lock/wait frames were negligible. The patched Nightly is
executing compute slowly rather than losing parallelism.

Both recipes below read `/tmp/bc-w-t12mm.nii` and `/tmp/bc-w-m16n.nii`, which
`make -C src test-wasm` writes — and which that gate has already asserted
bit-exact against `test/labels16_native_majority_wasm.nii.gz`, so the second doubles as the
expected output. Run it first.

Build a name-mapped module without changing the shipped Wasm:

```sh
make -C src wasm-model16chan18cls \
  WASM_DIR=/tmp/brainchopc-wasm-symbols WASM_CPU_SYMBOL_MAP=1
```

Note `WASM_DIR`: the symbol-mapped module differs from the shipped one in
121,043 bytes (index order and minified names, not code), so it must not be
built over `dist/wasm`.

Run only the page row in Playwright's Firefox:

```sh
BC_TEST_BROWSER=firefox BC_MT_MODES=page node test/wasm_mt_run.mjs \
  /tmp/brainchopc-wasm-symbols/brainchop-16chan18cls.js \
  /tmp/bc-w-t12mm.nii /tmp/bc-w-m16n.nii
```

Use installed Firefox as the product control:

```sh
BC_MT_MODES=release \
BC_FIREFOX_PATH=/Applications/Firefox.app/Contents/MacOS/firefox \
node test/wasm_mt_run.mjs \
  /tmp/brainchopc-wasm-symbols/brainchop-16chan18cls.js \
  /tmp/bc-w-t12mm.nii /tmp/bc-w-m16n.nii
```

That control reports the run and exits; it does NOT produce a capture, and this
is measured rather than assumed. On 153.0.1/macOS `window.close()` does not
close a window the script did not open, SIGINT is ignored, and the harness ends
the browser with SIGTERM — under which Gecko logs `AbnormalShutdown` and
`MOZ_PROFILER_SHUTDOWN` writes nothing.

For an actual capture the browser has to quit cleanly. Launch it with
`--marionette` and a `marionette.port` pref in the temp profile, then send
`[0,1,"Marionette:Quit",{}]` over that socket once the POST has arrived:
Firefox exited in 504 ms and wrote a 25.9 MB capture. Doing it by hand — start
the profiler in the browser UI, load the autorun page, stop and save — works
just as well and is why the harness does not carry the socket code.

The production changes this ruled out: x-block width 2 made the Nightly 291 s
and node 13.2 s; convolution-only `minsize` made node 66.9 s; global
`-fno-unroll-loops` made node 30.1 s. Every one of them made node worse for no
Nightly gain, which is the point: this browser is not the thing to tune against.

Relaxed SIMD is NOT on that list, and it would be wrong to read it as rejected:
it is worth 1.4× on node and Chromium and ships as `WASM_RELAXED_SIMD=1`. It is
off by default because fusion is the engine's choice and that costs the portable
bit-exact guarantee — not because it is slow. It moved the Nightly only
125.8 → 113.5 s, which is why it is mentioned here at all.
