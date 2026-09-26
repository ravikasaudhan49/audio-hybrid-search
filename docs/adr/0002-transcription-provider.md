# ADR-0002: Transcription: hosted ASR, Deepgram then AssemblyAI

**Status:** accepted · **Decided by:** project owner
**Context.** Results must name the speaker. The dev laptop cannot run Whisper-large + pyannote
well; the task allows hosted transcription. Gemini Live was considered and rejected (no reliable
word timestamps, paraphrasing, non-deterministic).

**Decision.** Deepgram Nova-3 was tested first: fast, accurate, 2 speakers per episode, but only
*numbered* speakers. After researching speaker naming (context inference, speaking order, voice
enrollment), AssemblyAI Speaker Identification was tested on a 3-minute clip with no names given
and returned both names correctly. AssemblyAI became the default; transcription was made
provider-neutral (a normalized transcript format), so Deepgram remains supported.

**Consequences.** Speaker names come from context and need a human check (artefacts such as
"Name - 1"). ASR errors propagate (e.g. *Krantz* / *Kranz*), mitigated by the fuzzy ranker and
keyterm prompting.
