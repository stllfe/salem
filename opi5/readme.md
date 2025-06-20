# Running LLMs on Orange Pi 5 Max SBC

`TBD...`

## Hardware

* SBC: Orange Pi 5 Max
* OS: Armbian Linux 25.8.0 orangepi5-max 6.1.115-vendor-rk35xx
* RKNPU driver: v0.9.8


## References

Most code borrowed and/or adopted from these repos:

- [original rockchip LLM examples](https://github.com/airockchip/rknn-llm/tree/main/examples/rkllm_server_demo)
- [rkllama](https://github.com/NotPunchnox/rkllama/tree/main)
- [c0zaut's RKLLM-Gradio](https://github.com/c0zaut/RKLLM-Gradio/tree/main)

Qwen2.5 models are also provided by c0zaut and can be found on [HuggingFace](https://huggingface.co/collections/c01zaut/qwen-25-rk3588-673962f99c1a0956f3435f6b).

## Testing

```bash
python test.py
```
