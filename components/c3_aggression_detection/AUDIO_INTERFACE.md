# Optional audio branch: implemented interface, unvalidated fusion

The working completion demonstration uses pose-only ST-GCN. It never silently
adds unrelated audio or invokes a sound-event download.

Existing code is preserved:

- `src/audio_events.py`: optional YAMNet sound classes for screaming, yelling,
  shouting, crying, cheering, laughter, music and speech.
- `src/features.py`: past-only FFT band features and bounded audio segments.
- `c3_model.py`: RiskNet visual/interaction/skeleton/audio encoders, modality
  masks and causal GRU. This checkpoint family differs from the current ST-GCN.
- `src/cache_dataset.py`, `src/train.py`, `src/live.py`: original AV cache,
  guarded supervised fitting and inference interfaces. Missing audio is masked.
- `src/audio_cue_audit.py`: association diagnostics for genuinely synchronized
  recordings with reviewed annotations.

There is no compatible trained and independently evaluated AV checkpoint in the
current model inventory. The new workbench therefore does not advertise audio
fusion. Existing optional interfaces and synthetic tests are implementation
evidence, not measured screaming detection or violence-prediction performance.

Before an AV experiment: obtain approved recordings with genuinely synchronized
audio/video; explicitly review visible cue/contact times independently; lock
source groups, splits and feature contracts; confirm real audio availability;
fit normalization on training only; select any thresholds on development only;
evaluate video-only and AV on the same eligible held-out source cohort. Use new
versioned outputs and preserve the locked test set. TensorFlow/YAMNet dependencies
are optional and were not installed or downloaded during this completion stage.

Audio event scores describe sound-class hypotheses. A scream or shout alone is
not evidence of violence, intent or correspondence to a visible person. The
existing RTSP ring buffer is best-effort and has not established timestamp-grade
AV synchronization. Live AV and real-time performance remain unvalidated.
