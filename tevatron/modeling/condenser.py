import os
import torch
from torch import nn
from transformers import BertConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertLayer
from transformers.modeling_outputs import BaseModelOutput

import logging

logger = logging.getLogger(__name__)


class Condenser(nn.Module):
    def __init__(
        self,
        lm: PreTrainedModel,
        n_head_layers: int = 2,
    ):
        super(Condenser, self).__init__()
        self.lm = lm
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.max_position_embeddings = 13
        self.c_head = nn.ModuleList(
            [BertLayer(bert_config.config) for _ in range(n_head_layers)]
        )
        self.c_head.apply(self.lm._init_weights)

    def forward(self, model_input):
        lm_out: BaseModelOutput = self.lm(
            **model_input,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = torch.stack([i[:, 0, :] for i in lm_out.hidden_states], dim=1)
        attention_mask = torch.ones(cls_hiddens.shape[:2], dtype=torch.int)

        for layer in self.c_head:
            layer_out = layer(
                cls_hiddens,
                attention_mask,
            )
            cls_hiddens = layer_out[0]

        return cls_hiddens[:, 0]

    @classmethod
    def from_pretrained(
            cls, n_head_layers, *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = cls(hf_model, n_head_layers)
        path = args[0]
        if os.path.exists(os.path.join(path, 'head.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'head.pt'), map_location="cpu")
            model.c_head.load_state_dict(model_dict)
        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig, n_head_layers
    ):
        hf_model = AutoModel.from_config(config)
        model = cls(hf_model,  n_head_layers)

        return model

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        model_dict = self.c_head.state_dict()
        torch.save(model_dict, os.path.join(output_dir, 'head.pt'))
