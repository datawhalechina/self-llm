# 2-MiniMax-M2 Model Architecture Analysis Blog

MiniMax-M2 was released on October 27, 2025. The model has 230B parameters, with only 10B active parameters.
A PR has been submitted to the transformers repository for this model, but it has not yet been merged. You can learn about the MiniMax-M2 model architecture by reading this PR.
Link: [https://github.com/huggingface/transformers/pull/42028](https://github.com/huggingface/transformers/pull/42028)

The model architecture diagram is as follows:
![MiniMax-M2-Architecture](images/01-01.png)
  
The model architecture of MiniMax-M2 is quite similar to Qwen3 MoE. The main differences are:
1. The expert routing weights are changed from direct Softmax to Sigmoid followed by division by the sum.
2. Jitter noise is added before the expert routing gate during training.
3. Expert weight score correction is used: `e_score_correction_bias`.



## MLP
The MLP module is mainly used for expert calculation after expert routing. The code is as follows:
```python
class MiniMaxM2MLP(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
```
Visualization is shown below:
![MiniMax-M2-MLP](images/01-02.png)
The hidden states pass through two linear layers simultaneously, one as projection and one as gate, then they are multiplied element-wise to get the new hidden states.

## Expert Routing

The expert module takes hidden_states, selected expert indices, and weights as input, and outputs the hidden states which are the weighted sum of the calculations from the selected experts. The code is as follows:
```python
class MiniMaxM2Experts(nn.ModuleList):
    ...
    def forward(...):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states

```


The expert routing module first uses a gating layer to calculate expert routing weights, and then uses the above expert module for calculation. If it is in training mode, jitter noise is added to prevent expert activations from being too concentrated. The code is as follows:
```python
class MiniMaxM2SparseMoeBlock(nn.Module):
    ...
    
    def __init__(self, config):
        ...
        # Define expert routing gate layer (Linear layer)
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        
        # Define the expert module mentioned above
        self.experts = MiniMaxM2Experts(config)
        
        # Define expert routing weight correction bias
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def route_tokens_to_experts(self, router_logits):
        ...
    
    def forward(...):
        ...
        # Add noise during training
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        ...
        # Calculate expert routing weights
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        
        # Calculate expert output
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        ...
        return hidden_states
```
![MiniMax-M2-Expert-Routing](./images/01-03.png)

