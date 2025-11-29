from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # 20251119 _代码增加：增加判断扩散模型是否启用
        self.diff_cfg = self.model_cfg.BACKBONE_3D.DIFF_MODEL.DIFF_LOSS_CFG

        self.debug_prefix = False

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn

        if self.diff_cfg.enable:
            diffusion_loss = batch_dict.get('diffusion_loss', 0.0)
            loss = (self.diff_cfg.origin_weight * loss +
                    self.diff_cfg.fuse_weight * diffusion_loss)

            if self.debug_prefix:
                print(f"  - loss计算最终统计: {loss}")
                print(f"  - loss计算扩散模型损失: {diffusion_loss}")
                print(f"  - loss计算原始权重: {self.diff_cfg.origin_weight}")
                print(f"  - loss计算扩散权重: {self.diff_cfg.fuse_weight}")

        return loss, tb_dict, disp_dict
