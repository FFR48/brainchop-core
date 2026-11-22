/*
 * Embed a model's weight file into the executable's read-only data.
 *
 * Compiled once per model, parameterised by two -D flags:
 *   BC_PAYLOAD_PATH  quoted path to model.bin, resolved by the assembler's
 *                    include path (-I), so it is relative to src/
 *   BC_PAYLOAD_SYM   symbol stem; exports <stem>_begin and <stem>_end
 *
 * .incbin copies the file verbatim at assembly time. That is the point: a
 * generated C array of the same bytes would be ~3.5 MB of source per model and
 * would slow every rebuild, and appending the payload to the linked binary
 * instead would invalidate the macOS code signature and force notarisation to
 * run after the append.
 *
 * The weights are embedded EXACTLY as they appear in model.bin. MindGrab's
 * TFJS-to-NIfTI spatial transpose still happens at load time: pre-transposing
 * here would duplicate weight-layout knowledge into the build for a saving
 * measured in microseconds.
 *
 * Uppercase .S so the C preprocessor runs first; token pasting below depends on
 * it. Mach-O prefixes C symbols with an underscore and spells the read-only
 * section differently from ELF, which is all the platform difference there is.
 *
 * MSVC has no .incbin. Windows would need a .rc resource, a generated array for
 * that toolchain only, or clang-cl (which does support it). Recorded as a reach
 * goal rather than solved here.
 */

#define BC_CAT2(a, b) a##b
#define BC_CAT(a, b) BC_CAT2(a, b)

/*
 * Only Mach-O prefixes C symbols with an underscore. wasm32 does not, despite
 * not being ELF either.
 */
#if defined(__APPLE__) && !defined(__wasm__)
#define BC_EXPORT(name) BC_CAT(_, name)
#else
#define BC_EXPORT(name) name
#endif

#define BC_BEGIN BC_EXPORT(BC_CAT(BC_PAYLOAD_SYM, _begin))
#define BC_END BC_EXPORT(BC_CAT(BC_PAYLOAD_SYM, _end))

/*
 * Three spellings of "read-only data". The wasm one needs the flags and type
 * fields spelled out; `.section .rodata` alone is rejected by the wasm
 * assembler.
 */
#if defined(__wasm__)
	.section .rodata,"",@
#elif defined(__APPLE__)
	.section __TEXT,__const
#else
	.section .rodata
#endif

	/* 64-byte alignment so the blob can be read as float32 directly, and so
	   the first weights land on a cache line. */
	.p2align 6
	.globl BC_BEGIN
BC_BEGIN:
	.incbin BC_PAYLOAD_PATH
/*
 * wasm REQUIRES a size on every data symbol -- omitting it is a hard backend
 * error, not a warning. `.-BC_BEGIN` is the standard idiom for "however much
 * .incbin just emitted", so the size stays derived from the file rather than
 * duplicated from the descriptor. Mach-O has no .size directive at all.
 */
#if defined(__wasm__) || defined(__ELF__)
	.size BC_BEGIN, .-BC_BEGIN
#endif
	.globl BC_END
BC_END:
#if defined(__wasm__) || defined(__ELF__)
	.size BC_END, 0
#endif

#if defined(__linux__)
	/* Mark the stack non-executable; without this a lone .S in the link makes
	   GNU ld fall back to an executable stack for the whole binary. */
	.section .note.GNU-stack,"",%progbits
#endif
