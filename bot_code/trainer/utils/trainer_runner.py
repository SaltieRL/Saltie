def run_trainer(trainer):
    print('setting up the trainer')
    trainer.setup_trainer()
    print('setting up the model')
    trainer.setup_model()
    print('running the trainer')
    trainer.run_trainer()
