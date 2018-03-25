from bot_code.trainer.base_classes.base_agent_trainer import BaseAgentTrainer


class BaseAgentTrainer(BaseAgentTrainer):
    action_handler = None  # Initialize as a bot_code.modelHelpers.actions.action_handler

    def instantiate_model(self, model_class):
        return model_class(self.sess,
                           self.action_handler.get_logit_size(),
                           action_handler=self.action_handler,
                           is_training=True,
                           optimizer=self.optimizer,
                           config_file=self.create_model_config())

