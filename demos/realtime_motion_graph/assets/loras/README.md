# LoRA weights

Drop `.safetensors` LoRA files here. The realtime motion-to-music
server reads its default LoRA from this directory via
`demos/realtime_motion_graph/server.py::LORA_PATH`, which is computed
relative to the server module so the same relative path works on any
host (local box, remote 5090, container, etc.).

Files in this directory are `.gitignore`-d by default because LoRA
weights are too large to commit. Sync them out-of-band:

```bash
# local -> remote 5090
rsync -avz demos/realtime_motion_graph/assets/loras/ \
    user@5090-host:/path/to/ACE-Step-1.5_alt/demos/realtime_motion_graph/assets/loras/
```

To change the default LoRA, either:
- rename your file to `deathsteap_1.safetensors` (no code change), or
- edit `LORA_PATH` at the top of `demos/realtime_motion_graph/server.py`
  to point at the new filename.

The server falls back to this path when the client sends
`lora: true` with no `lora_path`, which both the native thin client
and the web client do.
