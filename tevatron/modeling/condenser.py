import os
import torch
from torch import nn
from transformers import BertConfig, AutoModel, PretrainedConfig, PreTrainedModel
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
        self.c_head = nn.Linear(768, 384)

    def forward(self, model_input):
        lm_out: BaseModelOutput = self.lm(
            **model_input,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hidden_1 = self.c_head(lm_out.hidden_states[6][:, 0, :])
        cls_hidden_2 = self.c_head(lm_out.hidden_states[12][:, 0, :])
        cls_hidden = torch.concat([cls_hidden_1, cls_hidden_2], dim=1)

        return cls_hidden

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
