# 100% Pure C++ Server for Karthik Jayan Portfolio SLM (Zero Python Runtime)
FROM ghcr.io/ggml-org/llama.cpp:server

EXPOSE 8000

# Native C++ Server Runtime Configuration (Zero Python / Zero GIL)
ENV LLAMA_ARG_MODEL=/models/karthik_qwen1.5b_q8.gguf \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8000 \
    LLAMA_ARG_CTX_SIZE=2048 \
    LLAMA_ARG_THREADS=2 \
    LLAMA_ARG_API_KEY=kj_live_sec_789f2a4b1c \
    LLAMA_ARG_ALIAS=karthik-qwen2.5-1.5b \
    LLAMA_ARG_METRICS=1 \
    LLAMA_ARG_CONT_BATCHING=1

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/llama-server"]
