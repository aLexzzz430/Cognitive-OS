# Collection incidents and recovery

## TinyLlama pre-start listener probe, 2026-09-08

After the Granite response completed normally, the sequential collector downloaded and verified TinyLlama's weights. Its local port-availability probe then raised `OSError: [Errno 48] Address already in use` before starting the TinyLlama server or sending an inference request. A read of the previous server's local endpoint returned connection refused after that server had exited.

The probe was changed to use `SO_REUSEADDR`, matching a reusable server listener after a previous connection. This is consistent with residual TCP TIME_WAIT preventing the original non-reusable bind. It does not permit binding over a live conflicting listener, and no unrelated process was stopped. The restarted collector preserved the three existing responses, reused the verified TinyLlama weights and successfully started TinyLlama's first actual inference.

This was a collector transport/startup correction, not a model-response retry or a reason to select a different output. Sampling settings, model roster and prompt were unchanged. The local incident record retains the old and new process-session context outside the public packet.

## Phi-3 disk preflight, 2026-09-08

TinyLlama subsequently completed normally. Before downloading Phi-3, the collector stopped because free disk space was approximately 44 MB below the weight size plus its 750 MB headroom requirement. No Phi-3 inference had started. An unused, reproducible downloaded utility binary from a completed separate task was removed after checking it had no open handles; source artifacts and evidence were preserved. The existing four responses were preserved and Phi-3's first download began with the required headroom. This did not change models, quantization, sampling or prompt.
