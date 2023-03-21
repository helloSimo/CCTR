import os
import torch
from torch import nn
from transformers import BertConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertLayer
from transformers.modeling_outputs import BaseModelOutput

import logging

logger = logging.getLogger(__name__)
N_HEAD_LAYERS = 2


class Condenser(nn.Module):
    def __init__(
        self,
        lm: PreTrainedModel,
    ):
        super(Condenser, self).__init__()
        self.lm = lm
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.max_position_embeddings = 13
        self.c_head = nn.ModuleList(
            [BertLayer(bert_config) for _ in range(N_HEAD_LAYERS)]
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
        attention_mask = self.lm.get_extended_attention_mask(attention_mask=attention_mask,
                                                             input_shape=cls_hiddens.shape)

        for layer in self.c_head:
            layer_out = layer(
                cls_hiddens,
                attention_mask.to(model_input['attention_mask'].device),
            )
            cls_hiddens = layer_out[0]

        return cls_hiddens[:, 0]

    @classmethod
    def from_pretrained(
            cls, *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = cls(hf_model)
        path = args[0]
        if os.path.exists(os.path.join(path, 'head.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'head.pt'), map_location="cpu")
            model.c_head.load_state_dict(model_dict)
        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig
    ):
        hf_model = AutoModel.from_config(config)
        model = cls(hf_model)

        return model

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        model_dict = self.c_head.state_dict()
        torch.save(model_dict, os.path.join(output_dir, 'head.pt'))
