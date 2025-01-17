import torch.nn as nn
from typing import Any, Protocol


class HFGCProtocol(Protocol):
    """Protocol for modules that support gradient checkpointing with Hugging Face Transformers."""

    gradient_checkpointing: bool

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any]) -> None:
        """Enable gradient checkpointing.
        Args:
            gradient_checkpointing_kwargs: Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        
        See Also:
            - [Transformers Documentation - enable gradient checkpointing](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_enable)
            - [Transformers Source - enable implementation](https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/modeling_utils.py#L2521)
        """
        ...

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing.
        
        See Also:
            - [Transformers Documentation - disable gradient checkpointing](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_disable)
            - [Transformers Source - disable implementation](https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/modeling_utils.py#L2585)
        """
        ...


class MCGCProtocol(Protocol):
    """Protocol for modules that support gradient checkpointing with MosaicML Composer."""

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        """Determine if module should be checkpointed.
        
        See Also:
            - [Composer Documentation - FSDP auto wrap policy](https://github.com/mosaicml/composer/blob/7fa03545cc2025f256d914abc111a068d239d632/docs/source/notes/distributed_training.rst#composers-fsdp-auto-wrap-policy)
            - [MosaicML Examples - GPT implementation](https://github.com/mosaicml/examples/blob/6972fe3000d5a5480d8757ff710965514155e8db/llm/llm/gpt.py#L173-L175)
        """
        ...