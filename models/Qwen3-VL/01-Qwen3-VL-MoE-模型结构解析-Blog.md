# Qwen3-VL Model Architecture Analysis (DeepStack Analysis)

## Overview Comparison of Qwen3 VL MoE and Qwen2 VL
1. The text part of Qwen3 VL MoE adopts a Mixture of Experts (MoE) architecture, with a default of 60 experts. Only a portion of experts are activated each time (e.g., top-k=4), and it is also possible to control which layers use MoE; Qwen2 VL is a fully connected structure.
2. Qwen3 VL MoE uses DeepStack feature fusion. In Qwen2 VL, only the visual features of the last layer are extracted, and all visual information is injected at once in the text input layer; in Qwen3 VL MoE, the visual extraction part adopts multi-stage feature extraction, and corresponding visual features are gradually injected at different levels of text decoding.

> MoE is similar to Qwen3 MoE, please refer to: https://github.com/datawhalechina/self-llm/blob/master/models/Qwen3/01-Qwen3-%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E8%A7%A3%E6%9E%90-Blog.md

## Detailed Explanation of Qwen3 VL MoE Model DeepStack Architecture

> Original DeepStack Paper: https://arxiv.org/pdf/2406.04334
> This section mainly describes how the Qwen3 VL MoE model applies the DeepStack idea to the model.


### Feature Extraction
```python
# class Qwen3VLMoeModel
def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
    pixel_values = pixel_values.type(self.visual.dtype
    # Get image_embeds, deepstack_image_embeds
    image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_image_embeds
```

It can be seen that `image_embeds` and `deepstack_image_embeds` are obtained using the `self.visual` method. `image_embeds` are placed directly into input tokens, while `deepstack_image_embeds` are added layer by layer subsequently.
Next, let's look at how `self.visual` is implemented.
`self.visual` is a `Qwen3VLMoeVisionModel` object by default (can be replaced if modifying the model), the core code of forward is as follows:


```python
def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        ...
        # Define deepstack feature list
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            # Feature extraction
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs)
            # Collect deepstack features only at specific layers
            if layer_num in self.deepstack_visual_indexes:
                # Note that the features extracted here are not used directly, but each feature passes through a different merger layer
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                # Record deepstack_feature
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists
```

Feature Extraction Flowchart:
![Feature Extraction Flowchart](./images/01-01.png)

### Feature Injection
Code Explanation:

```python
# class Qwen3VLMoeTextModel
def forward(..., deepstack_visual_embeds) -> Union[tuple, BaseModelOutputWithPast]:
    ...
    # decoder layers
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(...)
        hidden_states = layer_outputs

        # Add visual features from deepstack_feature_list to the hidden states of the first few layers (depending on deepstack size)
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _deepstack_process(
    self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
):
    # Device alignment: ensure all tensors are on the same device
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    # Feature fusion: Residual connection, inject only at visual token positions, keeping original features of text tokens
    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states
```
It can be seen that in `Qwen3VLMoeTextModel`, decoding is performed layer by layer, where the first n (n is the number of deepstacks) decoder layers use the above `deepstack_image_embeds` for injection via residual connection respectively.
  
Visual Feature Injection Flowchart:  

![Visual Feature Injection Flowchart](./images/01-02.png)
