from turtle import forward
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import math
from transformers import DistilBertForQuestionAnswering


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout_prob):
        super().__init__()

        # self.n_heads = config.n_heads
        # self.dim = config.dim
        # self.dropout = nn.Dropout(p=config.attention_dropout)

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout_prob)

        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)


    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)
        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return context

class FeedForward(nn.Module):
    def __init__(self, dim_input: int = 768, dim_feedforward: int = 4*768):
        super().__init__()

        self.linear1 = nn.Linear(dim_input, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, dim_input)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class SwitchFeedForward(nn.Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(self, *,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_experts: int,
                 expert: FeedForward,
                 d_model: int):
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # make copies of the FFNs
        self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(n_experts)])
        # Routing layer and softmax
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """

        # Capture the shape to change shapes later
        seq_len, batch_size, d_model = x.shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, d_model)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.switch(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        final_output = final_output.view(seq_len, batch_size, d_model)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max


class SwitchTransformerLayer(nn.Module):
    """
    # Switch Transformer Block
    This is the same as [normal transformer block](../models.html#TransformerLayer)
    with handling extra outputs of switch feedforward module.
    """

    def __init__(self, *,
                 d_model: int,
                 attn: MultiHeadAttention,
                 feed_forward: SwitchFeedForward,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `attn` is the attention module
        * `feed_forward` is the feed forward module (which is the switching module in this case)
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the switching feed-forward network
        ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x, counts, route_prob, n_dropped, route_prob_max


class SwitchTransformer(nn.Module):
    """
    ## Switch Transformer
    """

    def __init__(self, layer, n_layers, n_experts, device, load_balancing_loss_ceof):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])
        self.qa_outputs = nn.Linear(768, 2)
        self.base_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        self.device = device
        self.load_balancing_loss_ceof = load_balancing_loss_ceof
        self.n_experts = n_experts # used to calculate lb loss

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def freeze_experts(self):
        # TODO: find how to freeze the experts in the SwitchTransformer
        pass

    #def forward(self, x: torch.Tensor, mask: torch.Tensor):
    def forward(self, batch):

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        start_positions = batch['start_positions'].to(self.device) if 'start_positions' in batch.keys() else None
        end_positions = batch['end_positions'].to(self.device) if 'end_positions' in batch.keys() else None

        outputs = self.base_model(input_ids, attention_mask=attention_mask, start_positions=None, end_positions=None, output_hidden_states=True)
        x = outputs.hidden_states[-1]
        # Run through each transformer layer
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []
        for layer in self.layers:
            x, f, p, n_d, p_max = layer(x=x, mask=attention_mask)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)
        # Finally, normalize the vectors
        output = self.norm(x)


        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

        loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(
                0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        counts = torch.stack(counts)
        route_prob = torch.stack(route_prob)
        route_prob_max = torch.stack(route_prob_max)
        total = counts.sum(dim=-1, keepdims=True)
        route_frac = counts / total
        route_prob = route_prob / total
        load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()
        loss = load_balancing_loss if loss is None else loss + self.load_balancing_loss_ceof * load_balancing_loss
        return start_logits, end_logits, loss



