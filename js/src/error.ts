/**
 * Every refusal this package makes carries a machine-readable `code` alongside
 * the prose, because the interesting failures here are ones a caller may want
 * to act on rather than merely display -- "this GPU is too small" is a
 * different situation from "you passed an option this model does not have".
 *
 * There is deliberately no fallback path behind any of these. A silently
 * downgraded segmentation is worse than a refusal -- cropping, for one, changes
 * the answer rather than the speed. So these are terminal, and they say which
 * limit was short and by how much.
 *
 * `auto` choosing webgpu -> webgl2 -> cpu is not a contradiction of that. It
 * happens BEFORE anything is fetched and only when the caller expressed no
 * preference; a NAMED backend that cannot run still throws. What the rule
 * forbids is answering a request for one thing with another.
 */
export type BrainchopErrorCode =
  | 'no-webgpu'
  | 'no-adapter'
  | 'device-too-small'
  | 'no-f16'
  | 'unsupported-option'
  | 'bad-input'
  | 'initialization-failed'
  | 'inference-failed'

export class BrainchopError extends Error {
  readonly code: BrainchopErrorCode
  /** The module's own output, when the failure came from inside it. */
  readonly log?: string[]

  constructor(code: BrainchopErrorCode, message: string, log?: string[]) {
    super(message)
    this.name = 'BrainchopError'
    this.code = code
    this.log = log
  }
}
